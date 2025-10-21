import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import zipfile
from datetime import datetime
from typing import Dict


def make_download_filename(idx: int, original_name: str) -> str:
    """ダウンロード用の安全なファイル名を作成する（拡張子はPNG固定）"""
    stem = original_name.rsplit(".", 1)[0]
    return f"processed_{idx}_{stem}.png"


def apply_image_process(
    image: Image.Image, process_type: str, params: Dict
) -> Image.Image:
    """
    画像に指定された処理を適用する

    Args:
        image: 入力PIL画像
        process_type: 処理名（セレクトボックスの値）
        params: スライダー等で指定されたパラメータ辞書

    Returns:
        処理後のPIL画像
    """
    processed_image = image.copy()

    if process_type == "グレースケール":
        processed_image = processed_image.convert("L")
    elif process_type == "ぼかし":
        processed_image = processed_image.filter(
            ImageFilter.GaussianBlur(params["blur_radius"])
        )
    elif process_type == "輪郭検出":
        processed_image = processed_image.filter(ImageFilter.FIND_EDGES)
    elif process_type == "シャープ化":
        processed_image = processed_image.filter(ImageFilter.SHARPEN)
    elif process_type == "明るさ調整":
        enhancer = ImageEnhance.Brightness(processed_image)
        processed_image = enhancer.enhance(params["brightness"])
    elif process_type == "コントラスト調整":
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(params["contrast"])
    elif process_type == "セピア":
        # グレースケール化した画像にセピアの色を付与（高速・簡潔）
        gray = processed_image.convert("L")
        processed_image = ImageOps.colorize(gray, black="#2e1f0f", white="#e0c9a6")
    elif process_type == "反転":
        processed_image = ImageOps.invert(processed_image.convert("RGB"))
    elif process_type == "左右反転":
        processed_image = ImageOps.mirror(processed_image)
    elif process_type == "上下反転":
        processed_image = ImageOps.flip(processed_image)
    elif process_type == "リサイズ":
        method = params.get("resize_method", "fit")
        # PIL 9+ では Image.Resampling.LANCZOS が推奨だが、後方互換で Image.LANCZOS を使用
        resample = Image.LANCZOS

        if method == "width":
            target_w = int(params.get("width", processed_image.width))
            ratio = target_w / processed_image.width
            target_h = max(1, int(processed_image.height * ratio))
            processed_image = processed_image.resize((target_w, target_h), resample)
        elif method == "height":
            target_h = int(params.get("height", processed_image.height))
            ratio = target_h / processed_image.height
            target_w = max(1, int(processed_image.width * ratio))
            processed_image = processed_image.resize((target_w, target_h), resample)
        elif method == "stretch":
            target_w = int(params.get("width", processed_image.width))
            target_h = int(params.get("height", processed_image.height))
            processed_image = processed_image.resize((target_w, target_h), resample)
        else:  # fit（内接）
            target_w = int(params.get("width", processed_image.width))
            target_h = int(params.get("height", processed_image.height))
            scale = min(
                target_w / processed_image.width, target_h / processed_image.height
            )
            new_w = max(1, int(processed_image.width * scale))
            new_h = max(1, int(processed_image.height * scale))
            processed_image = processed_image.resize((new_w, new_h), resample)
    elif process_type == "回転":
        processed_image = processed_image.rotate(params["angle"], expand=True)
    elif process_type == "エンボス":
        processed_image = processed_image.filter(ImageFilter.EMBOSS)
    elif process_type == "ポスタライズ":
        processed_image = ImageOps.posterize(
            processed_image.convert("RGB"), params["bits"]
        )
    elif process_type == "ソラリゼーション":
        processed_image = ImageOps.solarize(
            processed_image.convert("RGB"), params["threshold"]
        )

    return processed_image


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
            "リサイズ",
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

    # パラメータ設定
    params = {}
    if process_type == "ぼかし":
        params["blur_radius"] = st.slider("ぼかしの強さ", 0, 10, 2)
    elif process_type == "明るさ調整":
        params["brightness"] = st.slider("明るさ", 0.5, 2.0, 1.0, 0.1)
    elif process_type == "コントラスト調整":
        params["contrast"] = st.slider("コントラスト", 0.5, 2.0, 1.0, 0.1)
    elif process_type == "リサイズ":
        st.markdown("#### リサイズ設定")
        method_label = st.radio(
            "リサイズ方法",
            [
                "幅で指定",
                "高さで指定",
                "幅×高さ（比率維持）",
                "幅×高さ（そのまま）",
            ],
            horizontal=True,
        )
        method_map = {
            "幅で指定": "width",
            "高さで指定": "height",
            "幅×高さ（比率維持）": "fit",
            "幅×高さ（そのまま）": "stretch",
        }
        params["resize_method"] = method_map[method_label]
        if params["resize_method"] in ("width", "fit", "stretch"):
            params["width"] = st.number_input("幅 (px)", min_value=1, value=800)
        if params["resize_method"] in ("height", "fit", "stretch"):
            params["height"] = st.number_input("高さ (px)", min_value=1, value=600)
    elif process_type == "回転":
        params["angle"] = st.slider("回転角度", 0, 360, 90, 15)
    elif process_type == "ポスタライズ":
        params["bits"] = st.slider("階調レベル", 1, 8, 2)
    elif process_type == "ソラリゼーション":
        params["threshold"] = st.slider("しきい値", 0, 255, 128)

    st.divider()

    # 処理適用フラグと、処理後の画像を保存するリスト
    apply_proc = process_type != "なし"
    processed_images = []

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

        # 画像処理を適用
        processed_image = apply_image_process(image, process_type, params)

        with col2:
            st.write("**処理後の画像**")
            st.image(processed_image, use_container_width=True)

        # 処理後の画像を保存
        if apply_proc:
            processed_images.append((processed_image, uploaded_file.name))

        # 画像情報表示
        st.caption(
            f"サイズ: {image.size[0]} x {image.size[1]} ピクセル | フォーマット: {image.format} | モード: {image.mode}"
        )

        # ダウンロードボタン
        if apply_proc:
            buf = io.BytesIO()
            processed_image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label=f"📥 画像 {idx} をダウンロード",
                data=byte_im,
                file_name=make_download_filename(idx, uploaded_file.name),
                mime="image/png",
                key=f"download_{idx}",
            )

        if idx < len(uploaded_files):
            st.divider()

    # ZIPファイルでまとめてダウンロード
    if apply_proc and len(processed_images) > 1:
        st.divider()
        st.subheader("📦 まとめてダウンロード")

        # ZIPファイルを作成
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, (img, original_name) in enumerate(processed_images, 1):
                # 各画像をバイトストリームに保存
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                # ZIPファイルに追加
                filename = make_download_filename(idx, original_name)
                zip_file.writestr(filename, img_buffer.getvalue())

        zip_buffer.seek(0)

        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.download_button(
            label=f"🗜️ すべての画像をZIPでダウンロード ({len(processed_images)}枚)",
            data=zip_buffer.getvalue(),
            file_name=f"processed_images_{timestamp}.zip",
            mime="application/zip",
        )
