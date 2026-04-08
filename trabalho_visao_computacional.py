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

    # ── Matcher ───────────────────────────────────────────────
    if matcher == "BF":
        norm = cv2.NORM_L2 if detector == "SIFT" else cv2.NORM_HAMMING
        bf = cv2.BFMatcher(norm, crossCheck=False)
        raw = bf.knnMatch(des1, des2, k=2)
    else:  # FLANN
        if detector == "SIFT":
            index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
        else:
            index_params = dict(algorithm=6,            # FLANN_INDEX_LSH
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        des1 = des1.astype(np.float32) if detector == "SIFT" else des1
        des2 = des2.astype(np.float32) if detector == "SIFT" else des2
        raw = flann.knnMatch(des1, des2, k=2)

    # ── Filtro de Lowe ────────────────────────────────────────
    boas = []
    for par in raw:
        if len(par) == 2:
            m, n = par
            if m.distance < 0.75 * n.distance:
                boas.append(m)

    if len(boas) < 4:
        raise RuntimeError("Correspondências insuficientes para calcular homografia.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in boas]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in boas]).reshape(-1, 1, 2)

    # ── Homografia + Warping ──────────────────────────────────
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calcula cantos transformados de img2
    corners2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    corners2_t = cv2.perspectiveTransform(corners2, H)
    corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    todos = np.concatenate([corners1, corners2_t], axis=0)

    xmin, ymin = np.int32(todos.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(todos.max(axis=0).ravel() + 0.5)

    transf = np.array([[1, 0, -xmin],
                       [0, 1, -ymin],
                       [0, 0, 1]], dtype=np.float64)

    panorama = cv2.warpPerspective(img2, transf @ H, (xmax - xmin, ymax - ymin))
    panorama[-ymin:-ymin+h1, -xmin:-xmin+w1] = img1

    t1 = time.perf_counter()
    tempo_ms = (t1 - t0) * 1000

    return panorama, tempo_ms, len(boas)


# ─────────────────────────────────────────────────────────────
#  MÓDULO 2 – INTERFACE GESTUAL (Lucas-Kanade)
# ─────────────────────────────────────────────────────────────

def iniciar_interface_gestual(callback_log=None):
    """
    Abre a câmera e usa fluxo óptico de Lucas-Kanade para detectar
    gestos horizontais da mão → envia seta esquerda/direita via pyautogui.
    Pressione 'q' na janela para encerrar.
    """
    try:
        import pyautogui
    except ImportError:
        msg = "pyautogui não encontrado. Instale com: pip install pyautogui"
        if callback_log:
            callback_log(msg)
        messagebox.showerror("Dependência faltando", msg)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        msg = "Câmera não encontrada."
        if callback_log:
            callback_log(msg)
        messagebox.showerror("Câmera", msg)
        return

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(old_frame)
    LIMIAR_GESTO = 60      # pixels de deslocamento horizontal
    COOLDOWN = 1.0         # segundos entre gestos
    ultimo_gesto = 0.0

    if callback_log:
        callback_log("Interface gestual iniciada. Pressione 'q' para encerrar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None:
                bons_novos = p1[st == 1]
                bons_velhos = p0[st == 1]

                if len(bons_novos) > 0:
                    dx_medio = float(np.mean(bons_novos[:, 0] - bons_velhos[:, 0]))
                    agora = time.time()

                    if agora - ultimo_gesto > COOLDOWN:
                        if dx_medio > LIMIAR_GESTO:
                            pyautogui.press('right')
                            ultimo_gesto = agora
                            if callback_log:
                                callback_log(f"Gesto →  (dx={dx_medio:.1f}px) → Próximo slide")
                        elif dx_medio < -LIMIAR_GESTO:
                            pyautogui.press('left')
                            ultimo_gesto = agora
                            if callback_log:
                                callback_log(f"Gesto ←  (dx={dx_medio:.1f}px) → Slide anterior")

                for novo, velho in zip(bons_novos, bons_velhos):
                    a, b = novo.ravel().astype(int)
                    c, d = velho.ravel().astype(int)
                    mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

                old_gray = frame_gray.copy()
                p0 = bons_novos.reshape(-1, 1, 2)
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            mask = np.zeros_like(frame)

        saida = cv2.add(frame, mask)
        cv2.putText(saida, "Gesto: mova a mao para esq/dir  |  'q' para sair",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Interface Gestual – Lucas-Kanade", saida)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if callback_log:
        callback_log("Interface gestual encerrada.")


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
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Image as RLImage, Table, TableStyle)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        import tempfile, shutil
    except ImportError:
        raise ImportError("reportlab não encontrado. Instale: pip install reportlab")

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    titulo_style = ParagraphStyle('titulo', parent=styles['Title'],
                                  fontSize=16, spaceAfter=6, alignment=TA_CENTER)
    sub_style    = ParagraphStyle('sub', parent=styles['Heading2'],
                                  fontSize=12, spaceAfter=4)
    body_style   = ParagraphStyle('body', parent=styles['Normal'],
                                  fontSize=10, spaceAfter=6, alignment=TA_JUSTIFY)

    story = []
    largura_pag = A4[0] - 4*cm  # largura útil

    def np_para_rl(arr, largura=largura_pag, altura_max=7*cm):
        """Converte ndarray BGR → arquivo temporário → RLImage."""
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, arr)
        h, w = arr.shape[:2]
        ratio = w / h
        alt = min(altura_max, largura / ratio)
        lar = alt * ratio
        return RLImage(tmp.name, width=lar, height=alt), tmp.name

    # ── Capa ─────────────────────────────────────────────────
    story.append(Paragraph("Trabalho Prático 1 – Visão Computacional", titulo_style))
    story.append(Paragraph("UNICENTRO – Prof. Dr. Mauro Miazaki", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))

    # ── Imagens originais ─────────────────────────────────────
    story.append(Paragraph("1. Imagens Originais", sub_style))
    for label, path in [("Imagem 1 (esquerda)", img1_path),
                         ("Imagem 2 (direita)",  img2_path)]:
        story.append(Paragraph(label, body_style))
        img_arr = cv2.imread(path)
        rl_img, _ = np_para_rl(img_arr)
        story.append(rl_img)
        story.append(Spacer(1, 0.3*cm))

    # ── Panoramas ─────────────────────────────────────────────
    story.append(Paragraph("2. Resultados das Panorâmicas", sub_style))
    for r in resultados:
        story.append(Paragraph(f"Combinação: {r['label']}", body_style))
        rl_img, _ = np_para_rl(r['panorama'])
        story.append(rl_img)
        story.append(Paragraph(
            f"Tempo de processamento: {r['tempo_ms']:.1f} ms   |   "
            f"Correspondências utilizadas: {r['n_matches']}",
            body_style))
        story.append(Spacer(1, 0.3*cm))

    # ── Tabela comparativa ────────────────────────────────────
    story.append(Paragraph("3. Tabela Comparativa", sub_style))
    cabecalho = ["Combinação", "Tempo (ms)", "Correspondências", "Observação"]
    dados = [cabecalho]
    for r in resultados:
        dados.append([r['label'],
                      f"{r['tempo_ms']:.1f}",
                      str(r['n_matches']),
                      r.get('obs', '–')])

    tabela = Table(dados, colWidths=[4.5*cm, 2.5*cm, 3.5*cm, 5*cm])
    tabela.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ECF0F1'), colors.white]),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(tabela)
    story.append(Spacer(1, 0.5*cm))

    # ── Respostas às questões ─────────────────────────────────
    story.append(Paragraph("4. Análise e Respostas", sub_style))

    respostas = [
        ("4.1  Baixo consumo de CPU/bateria (aplicações móveis, robótica, sistemas embarcados)",
         "Para cenários com recursos computacionais limitados, a melhor escolha é a combinação "
         "ORB + BF. O ORB (Oriented FAST and Rotated BRIEF) é um detector/descritor binário "
         "projetado especificamente para ser rápido e leve, consumindo muito menos CPU do que o "
         "SIFT, que realiza cálculos em ponto flutuante de alta precisão. O matcher Brute-Force "
         "com norma Hamming é eficiente para descritores binários. Embora o FLANN possa ser mais "
         "rápido em grandes conjuntos, sua inicialização tem overhead que penaliza aplicações em "
         "tempo real com recursos escassos. Portanto: ORB + BF é a opção mais indicada."),

        ("4.2  Grandes conjuntos de dados (Big Data)",
         "Para grandes volumes de dados (muitos pontos de interesse ou muitas imagens), a combinação "
         "ORB + FLANN é a mais adequada. O FLANN (Fast Library for Approximate Nearest Neighbors) "
         "utiliza estruturas de dados otimizadas (como árvores KD ou LSH) para encontrar "
         "correspondências de forma aproximada, mas muito mais rápida que a busca exaustiva do BF. "
         "Com milhares de descritores, a diferença de desempenho é expressiva. O ORB mantém "
         "a leveza computacional dos descritores. Caso a precisão seja também importante em grandes "
         "conjuntos, SIFT + FLANN é uma alternativa com melhor qualidade, porém maior custo."),

        ("4.3  Melhor qualidade possível (fotos profissionais)",
         "Para a máxima qualidade da panorâmica, a combinação SIFT + FLANN é a recomendada. "
         "O SIFT (Scale-Invariant Feature Transform) detecta pontos de interesse altamente "
         "discriminativos, invariantes a escala e rotação, e gera descritores de 128 dimensões em "
         "ponto flutuante com excelente poder de distinção. O FLANN com índice KD-Tree garante "
         "buscas eficientes e precisas nesses descritores contínuos. A combinação resulta em maior "
         "número de correspondências corretas, homografia mais robusta e, consequentemente, "
         "costura (warping) com menos artefatos e melhor alinhamento visual. O custo computacional "
         "mais elevado é justificado em aplicações onde a qualidade é prioritária."),
    ]

    for titulo_r, texto_r in respostas:
        story.append(Paragraph(titulo_r, ParagraphStyle('q', parent=styles['Heading3'],
                                                         fontSize=10, spaceAfter=2)))
        story.append(Paragraph(texto_r, body_style))
        story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    return output_path


# ─────────────────────────────────────────────────────────────
#  INTERFACE GRÁFICA PRINCIPAL (Tkinter)
# ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visão Computacional – TP1 | UNICENTRO")
        self.geometry("900x680")
        self.configure(bg="#1E1E2E")
        self.resizable(True, True)

        self.img1_path = tk.StringVar()
        self.img2_path = tk.StringVar()
        self.resultados = []   # armazena resultados dos 4 panoramas

        self._build_ui()

    # ── Construção da UI ──────────────────────────────────────
    def _build_ui(self):
        CORES = dict(bg="#1E1E2E", painel="#2A2A3E", btn="#7C3AED",
                     btn_hover="#6D28D9", texto="#E2E8F0", accent="#A78BFA",
                     verde="#10B981", vermelho="#EF4444", amarelo="#F59E0B")
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

        # ── Notebook (abas) ───────────────────────────────────
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("TNotebook",        background=CORES['bg'])
        style.configure("TNotebook.Tab",    background=CORES['painel'],
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
        c = self._c
        frm = tk.Frame(nb, bg=c['bg'])
        nb.add(frm, text="  🖼  Panorâmica  ")

        # Seleção de imagens
        sel_frm = tk.Frame(frm, bg=c['painel'], pady=8, padx=10)
        sel_frm.pack(fill='x', padx=10, pady=(10, 4))
        tk.Label(sel_frm, text="Imagens de entrada", bg=c['painel'],
                 fg=c['accent'], font=("Helvetica", 11, "bold")).grid(
                 row=0, column=0, columnspan=3, sticky='w', pady=(0,6))

        for i, (lbl, var) in enumerate([("Imagem 1 (esquerda):", self.img1_path),
                                         ("Imagem 2 (direita):", self.img2_path)]):
            tk.Label(sel_frm, text=lbl, bg=c['painel'], fg=c['texto'],
                     font=("Helvetica", 10)).grid(row=i+1, column=0, sticky='w', padx=4)
            tk.Entry(sel_frm, textvariable=var, width=42,
                     bg="#3A3A5E", fg=c['texto'], insertbackground='white',
                     relief='flat').grid(row=i+1, column=1, padx=6)
            tk.Button(sel_frm, text="Procurar",
                      bg=c['btn'], fg='white', relief='flat', cursor='hand2',
                      command=lambda v=var: self._escolher_arquivo(v)
                      ).grid(row=i+1, column=2, padx=4)

        # Botões de combinação
        btn_frm = tk.Frame(frm, bg=c['bg'])
        btn_frm.pack(pady=8)
        combos = [("ORB + BF",    "ORB", "BF"),
                  ("ORB + FLANN", "ORB", "FLANN"),
                  ("SIFT + BF",   "SIFT","BF"),
                  ("SIFT + FLANN","SIFT","FLANN")]
        for label, det, mat in combos:
            tk.Button(btn_frm, text=label, width=14, pady=6,
                      bg=c['btn'], fg='white', relief='flat',
                      font=("Helvetica", 10, "bold"), cursor='hand2',
                      command=lambda d=det, m=mat, l=label:
                          self._rodar_panoramica(d, m, l)
                      ).pack(side='left', padx=6)

        tk.Button(btn_frm, text="▶  Rodar TODOS", width=16, pady=6,
                  bg=c['verde'], fg='white', relief='flat',
                  font=("Helvetica", 10, "bold"), cursor='hand2',
                  command=self._rodar_todos).pack(side='left', padx=6)

        # Log e preview
        mid = tk.Frame(frm, bg=c['bg'])
        mid.pack(fill='both', expand=True, padx=10, pady=4)

        self.log_pan = tk.Text(mid, height=6, bg="#12121E", fg=c['texto'],
                               font=("Courier", 9), relief='flat', state='disabled')
        self.log_pan.pack(fill='x')

        self.canvas_pan = tk.Label(mid, bg=c['bg'],
                                   text="[ prévia da panorâmica ]",
                                   fg="#555577", font=("Helvetica", 11))
        self.canvas_pan.pack(fill='both', expand=True, pady=4)

    # ── ABA GESTUAL ───────────────────────────────────────────
    def _aba_gestual(self, nb):
        c = self._c
        frm = tk.Frame(nb, bg=c['bg'])
        nb.add(frm, text="  🖐  Interface Gestual  ")

        tk.Label(frm,
                 text="Controle slides com gestos de mão\n"
                      "← Gesto para a esquerda → slide anterior\n"
                      "→ Gesto para a direita  → próximo slide",
                 bg=c['bg'], fg=c['texto'],
                 font=("Helvetica", 12), justify='center').pack(pady=30)

        tk.Button(frm, text="▶  Iniciar Interface Gestual",
                  bg=c['verde'], fg='white', relief='flat',
                  font=("Helvetica", 12, "bold"), cursor='hand2',
                  padx=20, pady=10,
                  command=self._iniciar_gestual).pack()

        tk.Label(frm,
                 text="(Pressione 'q' na janela da câmera para encerrar)",
                 bg=c['bg'], fg="#888899",
                 font=("Helvetica", 9)).pack(pady=6)

        self.log_gest = tk.Text(frm, height=8, bg="#12121E", fg=c['texto'],
                                font=("Courier", 9), relief='flat', state='disabled')
        self.log_gest.pack(fill='x', padx=20, pady=10)

    # ── ABA RELATÓRIO ─────────────────────────────────────────
    def _aba_relatorio(self, nb):
        c = self._c
        frm = tk.Frame(nb, bg=c['bg'])
        nb.add(frm, text="  📄  Relatório PDF  ")

        tk.Label(frm,
                 text="Gera PDF com imagens originais, os 4 panoramas e tabela comparativa.",
                 bg=c['bg'], fg=c['texto'], font=("Helvetica", 11)).pack(pady=20)

        tk.Button(frm, text="📥  Gerar Relatório PDF",
                  bg=self._c['btn'], fg='white', relief='flat',
                  font=("Helvetica", 12, "bold"), cursor='hand2',
                  padx=20, pady=10,
                  command=self._gerar_relatorio).pack()

        self.lbl_relatorio = tk.Label(frm, text="", bg=c['bg'],
                                      fg=c['verde'], font=("Helvetica", 10))
        self.lbl_relatorio.pack(pady=8)

    # ── Helpers ───────────────────────────────────────────────
    def _escolher_arquivo(self, var):
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("Todos", "*.*")])
        if path:
            var.set(path)

    def _log(self, widget, msg):
        widget.configure(state='normal')
        widget.insert('end', msg + "\n")
        widget.see('end')
        widget.configure(state='disabled')

    def _rodar_panoramica(self, detector, matcher, label):
        p1, p2 = self.img1_path.get(), self.img2_path.get()
        if not p1 or not p2:
            messagebox.showwarning("Atenção", "Selecione as duas imagens primeiro.")
            return

        self._log(self.log_pan, f"⏳ Processando {label}...")

        def tarefa():
            try:
                pan, t, n = criar_panoramica(p1, p2, detector, matcher)
                self._log(self.log_pan, f"✅ {label}: {t:.1f} ms | {n} correspondências")

                # Armazena resultado (substitui se já existir)
                self.resultados = [r for r in self.resultados if r['label'] != label]
                self.resultados.append(dict(label=label, panorama=pan,
                                            tempo_ms=t, n_matches=n))

                # Prévia
                self._mostrar_preview(pan)
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
        rgb = cv2.cvtColor(pan, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        MAX_W, MAX_H = 820, 260
        scale = min(MAX_W/w, MAX_H/h, 1.0)
        rgb_small = cv2.resize(rgb, (int(w*scale), int(h*scale)))
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_small))
        self.canvas_pan.configure(image=imgtk, text="")
        self.canvas_pan.image = imgtk

    def _iniciar_gestual(self):
        def tarefa():
            iniciar_interface_gestual(
                callback_log=lambda m: self._log(self.log_gest, m))
        threading.Thread(target=tarefa, daemon=True).start()

    def _gerar_relatorio(self):
        if not self.resultados:
            messagebox.showwarning("Atenção",
                "Gere ao menos uma panorâmica antes de criar o relatório.")
            return
        p1, p2 = self.img1_path.get(), self.img2_path.get()
        if not p1 or not p2:
            messagebox.showwarning("Atenção", "Selecione as duas imagens.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".pdf",
              filetypes=[("PDF", "*.pdf")], initialfile="relatorio_TP1.pdf")
        if not out:
            return
        try:
            gerar_relatorio(p1, p2, self.resultados, output_path=out)
            self.lbl_relatorio.configure(text=f"✅ Relatório salvo em:\n{out}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))


if __name__ == "__main__":
    app = App()
    app.mainloop()
