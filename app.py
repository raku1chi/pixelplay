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


def center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """画像中心から指定サイズで切り抜き（はみ出すサイズは画像に収まるよう制限）"""
    w, h = img.size
    tw = max(1, min(target_w, w))
    th = max(1, min(target_h, h))
    left = (w - tw) // 2
    top = (h - th) // 2
    right = left + tw
    bottom = top + th
    return img.crop((left, top, right, bottom))


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
    elif process_type == "切り抜き":
        method = params.get("crop_method", "square")
        if method == "square":
            size = int(params.get("size", min(processed_image.size)))
            processed_image = center_crop(processed_image, size, size)
        else:
            target_w = int(params.get("crop_width", processed_image.width))
            target_h = int(params.get("crop_height", processed_image.height))
            processed_image = center_crop(processed_image, target_w, target_h)
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


st.set_page_config(
    page_title="PixelPlay | かんたん画像加工", page_icon="🎨", layout="wide"
)

st.title("🎨 PixelPlay — かんたん画像加工")
st.markdown(
    """
    アップロードした画像にフィルター・サイズ変更・切り抜きなどをまとめて適用できます。

    - 複数画像に同じ加工を一括適用
    - プレビューを見ながら個別ダウンロード
    - 2枚以上ならZIPで一括ダウンロード

    ヒント: JPEG/PNGに対応しています。大きい画像は処理に少し時間がかかることがあります。
    """
)
st.caption("アップロードした画像は、このセッション内での処理にのみ使用されます。")

# 1. 画像をアップロード
st.markdown("### 1. 画像をアップロード")
uploaded_files = st.file_uploader(
    "ここに画像をドラッグ＆ドロップ、または選択（複数可）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="JPEG/PNGに対応。複数枚をまとめて選択できます。",
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)}枚の画像を読み込みました")

    # 2. 加工をえらぶ
    st.subheader("2. 加工をえらぶ")

    process_type = st.selectbox(
        "適用する加工",
        [
            "なし",
            "グレースケール",
            "ぼかし",
            "輪郭検出",
            "シャープ化",
            "明るさ調整",
            "コントラスト調整",
            "リサイズ",
            "切り抜き",
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
    elif process_type == "切り抜き":
        st.markdown("#### 切り抜き設定")
        crop_label = st.radio(
            "切り抜き方法",
            [
                "正方形（中央）",
                "幅×高さ（中央）",
            ],
            horizontal=True,
        )
        crop_map = {
            "正方形（中央）": "square",
            "幅×高さ（中央）": "rect",
        }
        params["crop_method"] = crop_map[crop_label]
        if params["crop_method"] == "square":
            params["size"] = st.number_input("一辺の長さ (px)", min_value=1, value=512)
        else:
            params["crop_width"] = st.number_input("幅 (px)", min_value=1, value=512)
            params["crop_height"] = st.number_input("高さ (px)", min_value=1, value=512)
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
            st.write("**加工後の画像**")
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
        st.subheader("📦 3. まとめてダウンロード")

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
