"""
Trabalho Prático 1 – Visão Computacional
UNICENTRO – Prof. Dr. Mauro Miazaki
Requisitos:
  (1) Interface interativa (GUI com Tkinter)
  (2) Aquisição de imagens (arquivos + câmera)
  (3a) Panorâmica: ORB+BF, ORB+FLANN, SIFT+BF, SIFT+FLANN
  (3b) Interface gestual com Lucas-Kanade + pyautogui
  (4) Geração de relatório PDF
"""

import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os

# ─────────────────────────────────────────────────────────────
#  MÓDULO 1 – PANORÂMICA
# ─────────────────────────────────────────────────────────────

def criar_panoramica(img1_path: str, img2_path: str, detector: str, matcher: str):
    """
    Gera imagem panorâmica a partir de duas imagens.

    Parameters
    ----------
    img1_path : str   – caminho da imagem da esquerda
    img2_path : str   – caminho da imagem da direita
    detector  : str   – 'SIFT' ou 'ORB'
    matcher   : str   – 'BF'  ou 'FLANN'

    Returns
    -------
    panorama  : np.ndarray  – imagem resultante
    tempo_ms  : float       – tempo de processamento em ms
    n_matches : int         – número de correspondências usadas
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Não foi possível carregar as imagens.")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    t0 = time.perf_counter()

    # ── Detector de pontos de interesse ──────────────────────
    if detector == "SIFT":
        det = cv2.SIFT_create()
    else:  # ORB
        det = cv2.ORB_create(nfeatures=2000)

    kp1, des1 = det.detectAndCompute(gray1, None)
    kp2, des2 = det.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        raise RuntimeError("Nenhum descritor encontrado nas imagens.")

    # ── Matcher ───────────────────────────────────────────────
    if matcher == "BF":
        norm = cv2.NORM_L2 if detector == "SIFT" else cv2.NORM_HAMMING
        bf  = cv2.BFMatcher(norm, crossCheck=False)
        raw = bf.knnMatch(des1, des2, k=2)
    else:  # FLANN
        if detector == "SIFT":
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
            index_params  = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
        else:
            des1 = des1.astype(np.uint8)
            des2 = des2.astype(np.uint8)
            index_params  = dict(algorithm=6,             # FLANN_INDEX_LSH
                                 table_number=6,
                                 key_size=12,
                                 multi_probe_level=1)
            search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw   = flann.knnMatch(des1, des2, k=2)

    # ── Filtro de Lowe ────────────────────────────────────────
    boas = []
    for par in raw:
        if len(par) == 2:
            m, n = par
            if m.distance < 0.75 * n.distance:
                boas.append(m)

    if len(boas) < 4:
        raise RuntimeError(
            f"Correspondências insuficientes ({len(boas)}) para calcular homografia. "
            "Tente imagens com mais sobreposição."
        )

    pts1 = np.float32([kp1[m.queryIdx].pt for m in boas]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in boas]).reshape(-1, 1, 2)

    # ── Homografia + Warping ──────────────────────────────────
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homografia não pôde ser calculada.")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2   = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    corners2_t = cv2.perspectiveTransform(corners2, H)
    corners1   = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    todos      = np.concatenate([corners1, corners2_t], axis=0)

    xmin, ymin = np.int32(todos.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(todos.max(axis=0).ravel() + 0.5)

    tx = max(0, -xmin)
    ty = max(0, -ymin)
    transf = np.array([[1, 0, tx],
                       [0, 1, ty],
                       [0, 0,  1]], dtype=np.float64)

    canvas_w = xmax - xmin
    canvas_h = ymax - ymin

    panorama = cv2.warpPerspective(img2, transf @ H, (canvas_w, canvas_h))

    y0, y1_ = ty, min(ty + h1, canvas_h)
    x0, x1_ = tx, min(tx + w1, canvas_w)
    panorama[y0:y1_, x0:x1_] = img1[:y1_-y0, :x1_-x0]

    t1       = time.perf_counter()
    tempo_ms = (t1 - t0) * 1000

    return panorama, tempo_ms, len(boas)


# ─────────────────────────────────────────────────────────────
#  MÓDULO 2 – INTERFACE GESTUAL (Lucas-Kanade) – embutida no Tkinter
# ─────────────────────────────────────────────────────────────

def _abrir_camera():
    """Tenta abrir a câmera nos índices 0 e 1."""
    for idx in range(2):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap
        cap.release()
    return None


class GestualController:
    """
    Gerencia o loop de câmera + Lucas-Kanade em thread separada.
    Envia cada frame processado via callback para o Tkinter (sem cv2.imshow).
    """

    def __init__(self, frame_cb, log_cb=None):
        """
        frame_cb : callable(np.ndarray BGR) – recebe cada frame processado
        log_cb   : callable(str)            – recebe mensagens de log
        """
        self._frame_cb = frame_cb
        self._log_cb   = log_cb
        self._rodando  = False
        self._thread   = None

    def iniciar(self):
        if self._rodando:
            return
        self._rodando = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def parar(self):
        self._rodando = False

    def esta_ativo(self):
        return self._rodando

    def _log(self, msg):
        if self._log_cb:
            self._log_cb(msg)

    # ── Segmentação de pele em HSV ────────────────────────────
    @staticmethod
    def _mascara_pele(frame):
        """
        Retorna máscara binária dos pixels com cor de pele humana.
        Usa dois intervalos HSV para cobrir tons claros e médios.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Intervalo principal (tons de pele universais)
        m1 = cv2.inRange(hsv, np.array([0,  20, 70]),
                              np.array([20, 255, 255]))
        # Intervalo complementar (tons mais avermelhados/bronzeados)
        m2 = cv2.inRange(hsv, np.array([170, 20, 70]),
                              np.array([180, 255, 255]))
        mascara = cv2.bitwise_or(m1, m2)

        # Remove ruído e preenche buracos
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel, iterations=2)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=3)
        mascara = cv2.GaussianBlur(mascara, (5, 5), 0)
        _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)
        return mascara

    @staticmethod
    def _maior_contorno_pele(mascara):
        """
        Retorna o maior contorno da máscara de pele (presumivelmente a mão).
        Ignora regiões muito pequenas (< 3000 px²) para descartar ruídos.
        """
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None
        maior = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(maior) < 3000:
            return None
        return maior

    def _loop(self):
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
        except ImportError:
            self._log("pyautogui não encontrado. Instale com: pip install pyautogui")
            self._rodando = False
            return

        cap = _abrir_camera()
        if cap is None:
            self._log("Câmera não encontrada (índices 0 e 1 testados).")
            self._rodando = False
            return

        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        feature_params = dict(maxCorners=60, qualityLevel=0.3,
                              minDistance=7, blockSize=7)

        ret, old_frame = cap.read()
        if not ret:
            self._log("Falha ao ler frame inicial da câmera.")
            cap.release()
            self._rodando = False
            return

        old_gray   = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        trilha     = np.zeros_like(old_frame)   # rastro do fluxo óptico
        p0         = None

        LIMIAR_GESTO = 60
        COOLDOWN     = 1.0
        ultimo_gesto = 0.0

        self._log("Câmera iniciada. Mostre a mão e mova para esq/dir.")
        self._log("Clique em '⏹ Parar' para encerrar.")

        while self._rodando:
            ret, frame = cap.read()
            if not ret:
                break

            # ── 1. Segmentação da mão por cor de pele ────────
            mascara_pele = self._mascara_pele(frame)
            contorno     = self._maior_contorno_pele(mascara_pele)

            # Máscara booleana restrita ao maior contorno (a mão)
            mascara_mao = np.zeros(mascara_pele.shape, dtype=np.uint8)
            if contorno is not None:
                cv2.drawContours(mascara_mao, [contorno], -1, 255, thickness=cv2.FILLED)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── 2. Lucas-Kanade apenas dentro da mão ─────────
            if p0 is not None and len(p0) > 0:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None, **lk_params
                )

                if p1 is not None and st is not None:
                    bons_novos  = p1[st == 1]
                    bons_velhos = p0[st == 1]

                    # Mantém só pontos que ainda estão dentro da mão
                    dentro = []
                    for pt in bons_novos:
                        x, y = int(pt[0]), int(pt[1])
                        if 0 <= y < mascara_mao.shape[0] and \
                           0 <= x < mascara_mao.shape[1] and \
                           mascara_mao[y, x] > 0:
                            dentro.append(True)
                        else:
                            dentro.append(False)
                    dentro = np.array(dentro)

                    if dentro.any():
                        bons_novos  = bons_novos[dentro]
                        bons_velhos = bons_velhos[dentro]

                        dx_medio = float(np.mean(bons_novos[:, 0] - bons_velhos[:, 0]))
                        agora    = time.time()

                        if agora - ultimo_gesto > COOLDOWN:
                            if dx_medio > LIMIAR_GESTO:
                                pyautogui.press('right')
                                ultimo_gesto = agora
                                self._log(f"Gesto →  (dx={dx_medio:.1f}px) → Próximo slide")
                            elif dx_medio < -LIMIAR_GESTO:
                                pyautogui.press('left')
                                ultimo_gesto = agora
                                self._log(f"Gesto ←  (dx={dx_medio:.1f}px) → Slide anterior")

                        for novo, velho in zip(bons_novos, bons_velhos):
                            a, b = novo.ravel().astype(int)
                            c, d = velho.ravel().astype(int)
                            trilha = cv2.line(trilha, (a, b), (c, d), (0, 255, 0), 2)
                            frame  = cv2.circle(frame, (a, b), 4, (0, 200, 255), -1)

                        old_gray = frame_gray.copy()
                        p0       = bons_novos.reshape(-1, 1, 2)
                    else:
                        # Pontos saíram da mão → reinicia
                        p0 = None
                else:
                    p0 = None
            else:
                trilha = np.zeros_like(frame)

            # Reinicia pontos: busca só dentro da máscara da mão
            if p0 is None or len(p0) == 0:
                if mascara_mao.any():
                    p0 = cv2.goodFeaturesToTrack(
                        frame_gray, mask=mascara_mao, **feature_params
                    )
                old_gray = frame_gray.copy()

            # ── 3. Desenha contorno verde da mão ─────────────
            saida = cv2.add(frame, trilha)
            if contorno is not None:
                cv2.drawContours(saida, [contorno], -1, (0, 255, 120), 2)

                # Retângulo delimitador + centróide
                x, y, w, h = cv2.boundingRect(contorno)
                cv2.rectangle(saida, (x, y), (x+w, y+h), (255, 200, 0), 1)
                M  = cv2.moments(contorno)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(saida, (cx, cy), 6, (0, 100, 255), -1)

            cv2.putText(saida, "Mova a mao: <- anterior  |  -> proximo",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # ← Envia frame para o Tkinter (sem cv2.imshow)
            self._frame_cb(saida)

        cap.release()
        self._rodando = False
        self._log("Interface gestual encerrada.")


# ─────────────────────────────────────────────────────────────
#  MÓDULO 3 – RELATÓRIO PDF
# ─────────────────────────────────────────────────────────────

def gerar_relatorio(img1_path, img2_path, resultados, output_path="relatorio.pdf"):
    """
    Gera PDF com imagens originais, os 4 panoramas e tabela comparativa.

    resultados : lista de dicts com chaves:
        label, panorama (np.ndarray), tempo_ms, n_matches
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Image as RLImage, Table, TableStyle
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        import tempfile as _tmp
    except ImportError:
        raise ImportError("reportlab não encontrado. Instale: pip install reportlab")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm
    )

    styles       = getSampleStyleSheet()
    titulo_style = ParagraphStyle('titulo', parent=styles['Title'],
                                  fontSize=16, spaceAfter=6, alignment=TA_CENTER)
    sub_style    = ParagraphStyle('sub',    parent=styles['Heading2'],
                                  fontSize=12, spaceAfter=4)
    body_style   = ParagraphStyle('body',   parent=styles['Normal'],
                                  fontSize=10, spaceAfter=6, alignment=TA_JUSTIFY)

    story       = []
    largura_pag = A4[0] - 4*cm
    tmp_files   = []

    def np_para_rl(arr, largura=largura_pag, altura_max=7*cm):
        t = _tmp.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(t.name, arr)
        t.close()
        tmp_files.append(t.name)
        h, w  = arr.shape[:2]
        ratio = w / h
        alt   = min(altura_max, largura / ratio)
        lar   = alt * ratio
        return RLImage(t.name, width=lar, height=alt)

    # ── Capa ─────────────────────────────────────────────────
    story.append(Paragraph("Trabalho Prático 1 – Visão Computacional", titulo_style))
    story.append(Paragraph("UNICENTRO – Prof. Dr. Mauro Miazaki", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))

    # ── Imagens originais ─────────────────────────────────────
    story.append(Paragraph("1. Imagens Originais", sub_style))
    for label, path in [("Imagem 1 (esquerda)", img1_path),
                         ("Imagem 2 (direita)",  img2_path)]:
        arr = cv2.imread(path)
        if arr is not None:
            story.append(Paragraph(label, body_style))
            story.append(np_para_rl(arr))
            story.append(Spacer(1, 0.3*cm))

    # ── Panoramas ─────────────────────────────────────────────
    story.append(Paragraph("2. Resultados das Panorâmicas", sub_style))
    for r in resultados:
        story.append(Paragraph(f"Combinação: {r['label']}", body_style))
        story.append(np_para_rl(r['panorama']))
        story.append(Paragraph(
            f"Tempo de processamento: {r['tempo_ms']:.1f} ms   |   "
            f"Correspondências utilizadas: {r['n_matches']}",
            body_style
        ))
        story.append(Spacer(1, 0.3*cm))

    # ── Tabela comparativa ────────────────────────────────────
    story.append(Paragraph("3. Tabela Comparativa", sub_style))

    obs_map = {
        "ORB + BF":    "Rápido; ideal para dispositivos com recursos limitados",
        "ORB + FLANN": "Bom desempenho em grandes conjuntos de dados",
        "SIFT + BF":   "Alta qualidade; custo computacional mais elevado",
        "SIFT + FLANN":"Melhor qualidade com busca eficiente (fotos profissionais)",
    }

    cabecalho = ["Combinação", "Tempo (ms)", "Correspondências", "Observação"]
    dados     = [cabecalho]
    for r in resultados:
        obs = r.get('obs') or obs_map.get(r['label'], '–')
        dados.append([r['label'], f"{r['tempo_ms']:.1f}", str(r['n_matches']), obs])

    tabela = Table(dados, colWidths=[4.5*cm, 2.5*cm, 3.5*cm, 5*cm])
    tabela.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR',      (0,0), (-1,0), colors.white),
        ('FONTNAME',       (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,-1), 9),
        ('ALIGN',          (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ECF0F1'), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, colors.grey),
        ('TOPPADDING',     (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',  (0,0), (-1,-1), 4),
    ]))
    story.append(tabela)
    story.append(Spacer(1, 0.5*cm))

    # ── Respostas às questões ─────────────────────────────────
    story.append(Paragraph("4. Análise e Respostas", sub_style))

    respostas = [
        (
            "4.1  Baixo consumo de CPU/bateria "
            "(aplicações móveis, robótica, sistemas embarcados)",
            "Para cenários com recursos computacionais limitados, a melhor escolha é a "
            "combinação ORB + BF. O ORB (Oriented FAST and Rotated BRIEF) é um "
            "detector/descritor binário projetado para ser rápido e leve, consumindo muito "
            "menos CPU do que o SIFT, que realiza cálculos em ponto flutuante de alta precisão. "
            "O matcher Brute-Force com norma Hamming é eficiente para descritores binários e não "
            "exige estruturas de dados adicionais. Embora o FLANN seja mais rápido em grandes "
            "conjuntos, sua inicialização adiciona overhead que penaliza aplicações em tempo real "
            "com recursos escassos. Portanto: ORB + BF é a opção mais indicada."
        ),
        (
            "4.2  Grandes conjuntos de dados (Big Data)",
            "Para grandes volumes de dados (muitos pontos de interesse ou muitas imagens), "
            "a combinação ORB + FLANN é a mais adequada. O FLANN (Fast Library for Approximate "
            "Nearest Neighbors) utiliza estruturas de dados otimizadas (LSH para descritores "
            "binários) para encontrar correspondências de forma aproximada, porém muito mais "
            "rápida que a busca exaustiva do BF. Com milhares de descritores, a diferença de "
            "desempenho é expressiva. O ORB mantém a leveza computacional. Caso a precisão seja "
            "prioritária, SIFT + FLANN é uma alternativa com melhor qualidade, mas maior custo."
        ),
        (
            "4.3  Melhor qualidade possível (fotos profissionais)",
            "Para a máxima qualidade da panorâmica, a combinação SIFT + FLANN é a recomendada. "
            "O SIFT (Scale-Invariant Feature Transform) detecta pontos de interesse altamente "
            "discriminativos, invariantes a escala e rotação, gerando descritores de 128 "
            "dimensões em ponto flutuante com excelente poder de distinção. O FLANN com índice "
            "KD-Tree garante buscas eficientes e precisas. A combinação resulta em maior número "
            "de correspondências corretas, homografia mais robusta e costura com menos artefatos "
            "e melhor alinhamento visual. O custo computacional mais elevado é justificado quando "
            "a qualidade é prioritária sobre a velocidade."
        ),
    ]

    for titulo_r, texto_r in respostas:
        story.append(Paragraph(
            titulo_r,
            ParagraphStyle('q', parent=styles['Heading3'], fontSize=10, spaceAfter=2)
        ))
        story.append(Paragraph(texto_r, body_style))
        story.append(Spacer(1, 0.3*cm))

    doc.build(story)

    for f in tmp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    return output_path


# ─────────────────────────────────────────────────────────────
#  INTERFACE GRÁFICA PRINCIPAL (Tkinter)
# ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visão Computacional – TP1 | UNICENTRO")
        self.geometry("960x720")
        self.configure(bg="#1E1E2E")
        self.resizable(True, True)

        self.img1_path     = tk.StringVar()
        self.img2_path     = tk.StringVar()
        self.resultados    = []
        self._gestual_ctrl = None   # instância de GestualController

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._ao_fechar)

    # ── Construção da UI ──────────────────────────────────────
    def _build_ui(self):
        CORES = dict(
            bg="#1E1E2E", painel="#2A2A3E",
            btn="#7C3AED", texto="#E2E8F0",
            accent="#A78BFA", verde="#10B981",
            vermelho="#EF4444", amarelo="#F59E0B"
        )
        self._c = CORES

        # ── Cabeçalho ─────────────────────────────────────────
        hdr = tk.Frame(self, bg=CORES['painel'], pady=10)
        hdr.pack(fill='x')
        tk.Label(hdr, text="🔭  Visão Computacional – Trabalho Prático 1",
                 bg=CORES['painel'], fg=CORES['accent'],
                 font=("Helvetica", 16, "bold")).pack()
        tk.Label(hdr, text="UNICENTRO  •  Prof. Dr. Mauro Miazaki",
                 bg=CORES['painel'], fg=CORES['texto'],
                 font=("Helvetica", 9)).pack()

        # ── Notebook ──────────────────────────────────────────
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("TNotebook",     background=CORES['bg'])
        style.configure("TNotebook.Tab", background=CORES['painel'],
                        foreground=CORES['texto'], padding=[12, 6],
                        font=("Helvetica", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", CORES['btn'])],
                  foreground=[("selected", "white")])

        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=10, pady=10)

        self._aba_panoramica(nb)
        self._aba_gestual(nb)
        self._aba_relatorio(nb)

    # ── ABA PANORÂMICA ────────────────────────────────────────
    def _aba_panoramica(self, nb):
        c   = self._c
        frm = tk.Frame(nb, bg=c['bg'])
        nb.add(frm, text="  🖼  Panorâmica  ")

        # Seleção de imagens
        sel = tk.Frame(frm, bg=c['painel'], pady=8, padx=10)
        sel.pack(fill='x', padx=10, pady=(10, 4))
        tk.Label(sel, text="Imagens de entrada", bg=c['painel'], fg=c['accent'],
                 font=("Helvetica", 11, "bold")).grid(
                 row=0, column=0, columnspan=3, sticky='w', pady=(0,6))

        for i, (lbl, var) in enumerate([("Imagem 1 (esquerda):", self.img1_path),
                                         ("Imagem 2 (direita):",  self.img2_path)]):
            tk.Label(sel, text=lbl, bg=c['painel'], fg=c['texto'],
                     font=("Helvetica", 10)).grid(row=i+1, column=0, sticky='w', padx=4)
            tk.Entry(sel, textvariable=var, width=46,
                     bg="#3A3A5E", fg=c['texto'],
                     insertbackground='white', relief='flat'
                     ).grid(row=i+1, column=1, padx=6)
            tk.Button(sel, text="Procurar",
                      bg=c['btn'], fg='white', relief='flat', cursor='hand2',
                      command=lambda v=var: self._escolher_arquivo(v)
                      ).grid(row=i+1, column=2, padx=4)

        # Botões de combinação
        bf = tk.Frame(frm, bg=c['bg'])
        bf.pack(pady=8)
        combos = [("ORB + BF","ORB","BF"), ("ORB + FLANN","ORB","FLANN"),
                  ("SIFT + BF","SIFT","BF"), ("SIFT + FLANN","SIFT","FLANN")]
        for label, det, mat in combos:
            tk.Button(bf, text=label, width=14, pady=6,
                      bg=c['btn'], fg='white', relief='flat',
                      font=("Helvetica", 10, "bold"), cursor='hand2',
                      command=lambda d=det, m=mat, l=label:
                          self._rodar_panoramica(d, m, l)
                      ).pack(side='left', padx=6)
        tk.Button(bf, text="▶  Rodar TODOS", width=16, pady=6,
                  bg=c['verde'], fg='white', relief='flat',
                  font=("Helvetica", 10, "bold"), cursor='hand2',
                  command=self._rodar_todos).pack(side='left', padx=6)

        # Log + preview
        mid = tk.Frame(frm, bg=c['bg'])
        mid.pack(fill='both', expand=True, padx=10, pady=4)
        self.log_pan = tk.Text(mid, height=6, bg="#12121E", fg=c['texto'],
                               font=("Courier", 9), relief='flat', state='disabled')
        self.log_pan.pack(fill='x')
        self.canvas_pan = tk.Label(mid, bg=c['bg'],
                                   text="[ prévia da panorâmica aparecerá aqui ]",
                                   fg="#555577", font=("Helvetica", 11))
        self.canvas_pan.pack(fill='both', expand=True, pady=4)

    # ── ABA GESTUAL ───────────────────────────────────────────
    def _aba_gestual(self, nb):
        c   = self._c
        frm = tk.Frame(nb, bg=c['bg'])
        nb.add(frm, text="  🖐  Interface Gestual  ")

        # Painel de instruções
        info = tk.Frame(frm, bg=c['painel'], pady=10, padx=20)
        info.pack(fill='x', padx=10, pady=(10, 4))
        tk.Label(info,
                 text=(
                     "Controle slides com gestos de mão\n"
                     "←  Mão para a esquerda  →  slide anterior   |   "
                     "→  Mão para a direita  →  próximo slide\n"
                     "Algoritmo: Fluxo Óptico Esparso de Lucas-Kanade  •  pyautogui"
                 ),
                 bg=c['painel'], fg=c['texto'],
                 font=("Helvetica", 10), justify='center').pack()

        # Botões iniciar / parar
        botoes = tk.Frame(frm, bg=c['bg'])
        botoes.pack(pady=8)

        self.btn_iniciar = tk.Button(
            botoes, text="▶  Iniciar Câmera",
            bg=c['verde'], fg='white', relief='flat',
            font=("Helvetica", 11, "bold"), cursor='hand2',
            padx=18, pady=8,
            command=self._iniciar_gestual
        )
        self.btn_iniciar.pack(side='left', padx=8)

        self.btn_parar = tk.Button(
            botoes, text="⏹  Parar",
            bg=c['vermelho'], fg='white', relief='flat',
            font=("Helvetica", 11, "bold"), cursor='hand2',
            padx=18, pady=8, state='disabled',
            command=self._parar_gestual
        )
        self.btn_parar.pack(side='left', padx=8)

        # ── Preview da câmera embutido no Tkinter ─────────────
        self.lbl_camera = tk.Label(frm, bg='black',
                                   text="[ câmera aparecerá aqui ]",
                                   fg="#555577", font=("Helvetica", 11))
        self.lbl_camera.pack(fill='both', expand=True, padx=10, pady=4)

        # Log de gestos
        self.log_gest = tk.Text(frm, height=5, bg="#12121E", fg=c['texto'],
                                font=("Courier", 9), relief='flat', state='disabled')
        self.log_gest.pack(fill='x', padx=10, pady=(0, 8))

    # ── ABA RELATÓRIO ─────────────────────────────────────────
    def _aba_relatorio(self, nb):
        c   = self._c
        frm = tk.Frame(nb, bg=c['bg'])
        nb.add(frm, text="  📄  Relatório PDF  ")

        tk.Label(frm,
                 text=(
                     "Gera PDF com:\n"
                     "  • Imagens originais\n"
                     "  • Os 4 resultados de panorâmica\n"
                     "  • Tabela comparativa de tempo e correspondências\n"
                     "  • Análise e respostas às questões do enunciado"
                 ),
                 bg=c['bg'], fg=c['texto'],
                 font=("Helvetica", 11), justify='left').pack(pady=20)

        tk.Label(frm,
                 text="⚠  Gere ao menos uma panorâmica antes de criar o relatório.",
                 bg=c['bg'], fg=c['amarelo'], font=("Helvetica", 9)).pack()

        tk.Button(frm, text="📥  Gerar Relatório PDF",
                  bg=c['btn'], fg='white', relief='flat',
                  font=("Helvetica", 12, "bold"), cursor='hand2',
                  padx=20, pady=10,
                  command=self._gerar_relatorio).pack(pady=10)

        self.lbl_relatorio = tk.Label(frm, text="", bg=c['bg'],
                                      fg=c['verde'], font=("Helvetica", 10),
                                      wraplength=600)
        self.lbl_relatorio.pack(pady=8)

    # ── Helpers gerais ────────────────────────────────────────
    def _escolher_arquivo(self, var):
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("Todos", "*.*")])
        if path:
            var.set(path)

    def _log(self, widget, msg):
        def _ins():
            widget.configure(state='normal')
            widget.insert('end', msg + "\n")
            widget.see('end')
            widget.configure(state='disabled')
        self.after(0, _ins)

    # ── Panorâmica ────────────────────────────────────────────
    def _rodar_panoramica(self, detector, matcher, label):
        p1, p2 = self.img1_path.get(), self.img2_path.get()
        if not p1 or not p2:
            messagebox.showwarning("Atenção", "Selecione as duas imagens primeiro.")
            return
        self._log(self.log_pan, f"⏳ Processando {label}...")

        def tarefa():
            try:
                pan, t, n = criar_panoramica(p1, p2, detector, matcher)
                self._log(self.log_pan,
                          f"✅ {label}: {t:.1f} ms | {n} correspondências")
                self.resultados = [r for r in self.resultados if r['label'] != label]
                self.resultados.append(
                    dict(label=label, panorama=pan, tempo_ms=t, n_matches=n))
                self.after(0, lambda: self._mostrar_preview(pan))
            except Exception as e:
                self._log(self.log_pan, f"❌ Erro ({label}): {e}")

        threading.Thread(target=tarefa, daemon=True).start()

    def _rodar_todos(self):
        for label, det, mat in [("ORB + BF","ORB","BF"),
                                  ("ORB + FLANN","ORB","FLANN"),
                                  ("SIFT + BF","SIFT","BF"),
                                  ("SIFT + FLANN","SIFT","FLANN")]:
            self._rodar_panoramica(det, mat, label)

    def _mostrar_preview(self, pan):
        rgb   = cv2.cvtColor(pan, cv2.COLOR_BGR2RGB)
        h, w  = rgb.shape[:2]
        scale = min(840/w, 280/h, 1.0)
        small = cv2.resize(rgb, (int(w*scale), int(h*scale)))
        imgtk = ImageTk.PhotoImage(Image.fromarray(small))
        self.canvas_pan.configure(image=imgtk, text="")
        self.canvas_pan.image = imgtk

    # ── Interface gestual ─────────────────────────────────────
    def _iniciar_gestual(self):
        if self._gestual_ctrl and self._gestual_ctrl.esta_ativo():
            return

        self.btn_iniciar.configure(state='disabled')
        self.btn_parar.configure(state='normal')
        self.lbl_camera.configure(text="Aguardando câmera...", image='')

        self._gestual_ctrl = GestualController(
            frame_cb=self._atualizar_camera,
            log_cb=lambda m: self._log(self.log_gest, m)
        )
        self._gestual_ctrl.iniciar()

    def _parar_gestual(self):
        if self._gestual_ctrl:
            self._gestual_ctrl.parar()
        self.btn_iniciar.configure(state='normal')
        self.btn_parar.configure(state='disabled')
        self.lbl_camera.configure(image='',
                                  text="[ câmera encerrada ]",
                                  fg="#555577")
        self.lbl_camera.image = None

    def _atualizar_camera(self, frame_bgr):
        """Recebe frame BGR da thread e atualiza o Label da câmera (thread-safe)."""
        def _render():
            if not (self._gestual_ctrl and self._gestual_ctrl.esta_ativo()):
                return
            rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w  = rgb.shape[:2]
            # Escala para caber no label (máx 880 × 380 px)
            scale = min(880/w, 380/h, 1.0)
            small = cv2.resize(rgb, (int(w*scale), int(h*scale)))
            imgtk = ImageTk.PhotoImage(Image.fromarray(small))
            self.lbl_camera.configure(image=imgtk, text="")
            self.lbl_camera.image = imgtk   # evita garbage collection
        self.after(0, _render)

    # ── Relatório ─────────────────────────────────────────────
    def _gerar_relatorio(self):
        if not self.resultados:
            messagebox.showwarning("Atenção",
                "Gere ao menos uma panorâmica antes de criar o relatório.")
            return
        p1, p2 = self.img1_path.get(), self.img2_path.get()
        if not p1 or not p2:
            messagebox.showwarning("Atenção", "Selecione as duas imagens.")
            return
        out = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF", "*.pdf")],
            initialfile="relatorio_TP1.pdf")
        if not out:
            return

        self.lbl_relatorio.configure(text="⏳ Gerando relatório...",
                                     fg=self._c['amarelo'])

        def tarefa():
            try:
                gerar_relatorio(p1, p2, self.resultados, output_path=out)
                self.after(0, lambda: self.lbl_relatorio.configure(
                    text=f"✅ Relatório salvo em:\n{out}",
                    fg=self._c['verde']))
            except Exception as e:
                self.after(0, lambda: (
                    self.lbl_relatorio.configure(
                        text=f"❌ Erro: {e}", fg=self._c['vermelho']),
                    messagebox.showerror("Erro", str(e))
                ))

        threading.Thread(target=tarefa, daemon=True).start()

    # ── Fechamento seguro ─────────────────────────────────────
    def _ao_fechar(self):
        if self._gestual_ctrl:
            self._gestual_ctrl.parar()
        self.destroy()


# ─────────────────────────────────────────────────────────────
#  PONTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()