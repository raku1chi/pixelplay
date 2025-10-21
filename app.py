import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import io

st.title("画像処理アプリ")

# 画像アップロード
uploaded_file = st.file_uploader(
    "画像をアップロードしてください", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 画像を読み込み
    image = Image.open(uploaded_file)

    # 2カラムレイアウト
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("元の画像")
        st.image(image, use_container_width=True)

    # 処理オプション
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
        ],
    )

    processed_image = image.copy()

    if process_type == "グレースケール":
        processed_image = processed_image.convert("L")
    elif process_type == "ぼかし":
        blur_radius = st.slider("ぼかしの強さ", 0, 10, 2)
        processed_image = processed_image.filter(ImageFilter.GaussianBlur(blur_radius))
    elif process_type == "輪郭検出":
        processed_image = processed_image.filter(ImageFilter.FIND_EDGES)
    elif process_type == "シャープ化":
        processed_image = processed_image.filter(ImageFilter.SHARPEN)
    elif process_type == "明るさ調整":
        brightness = st.slider("明るさ", 0.5, 2.0, 1.0, 0.1)
        enhancer = ImageEnhance.Brightness(processed_image)
        processed_image = enhancer.enhance(brightness)
    elif process_type == "コントラスト調整":
        contrast = st.slider("コントラスト", 0.5, 2.0, 1.0, 0.1)
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(contrast)

    with col2:
        st.subheader("処理後の画像")
        st.image(processed_image, use_container_width=True)

    # 画像情報表示
    st.subheader("画像情報")
    st.write(f"サイズ: {image.size[0]} x {image.size[1]} ピクセル")
    st.write(f"フォーマット: {image.format}")
    st.write(f"モード: {image.mode}")

    # ダウンロードボタン
    if process_type != "なし":
        buf = io.BytesIO()
        processed_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="処理後の画像をダウンロード",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png",
        )
