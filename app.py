import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import numpy as np

st.title("画像処理アプリ")

# 画像アップロード
uploaded_files = st.file_uploader(
    "画像をアップロードしてください（複数選択可）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)}枚の画像がアップロードされました")

    # 処理オプションを先に選択
    st.subheader("画像処理オプション")

    process_type = st.selectbox(
        "処理を選択してください",
        [
            "なし",
            "グレースケール",
            "ぼかし",
            "輪郭検出",
            "シャープ化",
            "明るさ調整",
            "コントラスト調整",
            "セピア",
            "反転",
            "左右反転",
            "上下反転",
            "回転",
            "エンボス",
            "ポスタライズ",
            "ソラリゼーション",
        ],
    )

    # パラメータ用の変数を事前に定義
    blur_radius = None
    brightness = None
    contrast = None
    angle = None
    bits = None
    threshold = None

    # パラメータ設定
    if process_type == "ぼかし":
        blur_radius = st.slider("ぼかしの強さ", 0, 10, 2)
    elif process_type == "明るさ調整":
        brightness = st.slider("明るさ", 0.5, 2.0, 1.0, 0.1)
    elif process_type == "コントラスト調整":
        contrast = st.slider("コントラスト", 0.5, 2.0, 1.0, 0.1)
    elif process_type == "回転":
        angle = st.slider("回転角度", 0, 360, 90, 15)
    elif process_type == "ポスタライズ":
        bits = st.slider("階調レベル", 1, 8, 2)
    elif process_type == "ソラリゼーション":
        threshold = st.slider("しきい値", 0, 255, 128)

    st.divider()

    # 各画像を処理
    for idx, uploaded_file in enumerate(uploaded_files, 1):
        st.subheader(f"画像 {idx}: {uploaded_file.name}")

        # 画像を読み込み
        image = Image.open(uploaded_file)

        # 2カラムレイアウト
        col1, col2 = st.columns(2)

        with col1:
            st.write("**元の画像**")
            st.image(image, use_container_width=True)

        processed_image = image.copy()

        if process_type == "グレースケール":
            processed_image = processed_image.convert("L")
        elif process_type == "ぼかし":
            processed_image = processed_image.filter(
                ImageFilter.GaussianBlur(blur_radius)
            )
        elif process_type == "輪郭検出":
            processed_image = processed_image.filter(ImageFilter.FIND_EDGES)
        elif process_type == "シャープ化":
            processed_image = processed_image.filter(ImageFilter.SHARPEN)
        elif process_type == "明るさ調整":
            enhancer = ImageEnhance.Brightness(processed_image)
            processed_image = enhancer.enhance(brightness)
        elif process_type == "コントラスト調整":
            enhancer = ImageEnhance.Contrast(processed_image)
            processed_image = enhancer.enhance(contrast)
        elif process_type == "セピア":
            # セピア調に変換
            processed_image = processed_image.convert("RGB")
            width, height = processed_image.size
            pixels = processed_image.load()
            for py in range(height):
                for px in range(width):
                    r, g, b = processed_image.getpixel((px, py))
                    tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                    tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                    tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                    pixels[px, py] = (min(tr, 255), min(tg, 255), min(tb, 255))
        elif process_type == "反転":
            processed_image = ImageOps.invert(processed_image.convert("RGB"))
        elif process_type == "左右反転":
            processed_image = ImageOps.mirror(processed_image)
        elif process_type == "上下反転":
            processed_image = ImageOps.flip(processed_image)
        elif process_type == "回転":
            processed_image = processed_image.rotate(angle, expand=True)
        elif process_type == "エンボス":
            processed_image = processed_image.filter(ImageFilter.EMBOSS)
        elif process_type == "ポスタライズ":
            processed_image = ImageOps.posterize(processed_image.convert("RGB"), bits)
        elif process_type == "ソラリゼーション":
            processed_image = ImageOps.solarize(
                processed_image.convert("RGB"), threshold
            )

        with col2:
            st.write("**処理後の画像**")
            st.image(processed_image, use_container_width=True)

        # 画像情報表示
        st.caption(
            f"サイズ: {image.size[0]} x {image.size[1]} ピクセル | フォーマット: {image.format} | モード: {image.mode}"
        )

        # ダウンロードボタン
        if process_type != "なし":
            buf = io.BytesIO()
            processed_image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label=f"📥 画像 {idx} をダウンロード",
                data=byte_im,
                file_name=f"processed_{idx}_{uploaded_file.name}",
                mime="image/png",
                key=f"download_{idx}",
            )

        if idx < len(uploaded_files):
            st.divider()
