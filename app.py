import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import os
from pathlib import Path
import io
import zipfile
from datetime import datetime
from typing import Dict


def make_download_filename(idx: int, original_name: str, ext: str) -> str:
    """ダウンロード用の安全なファイル名を作成する（拡張子を指定）"""
    stem = original_name.rsplit(".", 1)[0]
    return f"processed_{idx}_{stem}.{ext}"


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


def prepare_download_bytes(
    img: Image.Image,
    fmt: str,
    jpeg_quality: int | None = None,
    exif_bytes: bytes | None = None,
):
    """出力形式に応じて画像をエンコードして、(bytes, ext, mime) を返す。

    exif_bytes は JPEG のみ有効。None の場合は EXIF なしで保存。
    """
    fmt = fmt.upper()
    if fmt not in ("PNG", "JPEG"):
        fmt = "PNG"
    ext = "png" if fmt == "PNG" else "jpg"
    mime = "image/png" if fmt == "PNG" else "image/jpeg"

    out = img
    if fmt == "JPEG" and out.mode not in ("RGB", "L"):
        # JPEGはアルファをサポートしないため安全にRGBへ
        out = out.convert("RGB")

    buf = io.BytesIO()
    save_kwargs = {"format": fmt}
    if fmt == "JPEG" and jpeg_quality is not None:
        save_kwargs.update({"quality": int(jpeg_quality), "optimize": True})
    if fmt == "JPEG" and exif_bytes:
        save_kwargs.update({"exif": exif_bytes})
    out.save(buf, **save_kwargs)
    return buf.getvalue(), ext, mime


def build_exif_bytes(src: Image.Image, policy: str) -> bytes | None:
    """元画像のEXIFから、方針に応じてバイト列を返す。

    policy: 'keep' | 'strip_gps' | 'strip_all'
    Orientation は自動回転を行うため 1 に正規化します。
    """
    try:
        exif = src.getexif()
    except Exception:
        exif = None
    if not exif or len(exif) == 0:
        return None

    ORIENTATION = 274  # Orientation
    GPSINFO = 34853  # GPSInfo

    # Orientation を 1 に正規化（表示側の二重回転を防止）
    if ORIENTATION in exif:
        exif[ORIENTATION] = 1

    if policy == "strip_all":
        return None
    if policy == "strip_gps" and GPSINFO in exif:
        try:
            del exif[GPSINFO]
        except Exception:
            pass
    # keep または strip_gps 後のバイト列
    try:
        return exif.tobytes()
    except Exception:
        return None


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


# サイドバー（設定）
sb = st.sidebar
sb.header("設定")

# 1. 画像をアップロード
st.markdown("### 1. 画像をアップロード")
uploaded_files = st.file_uploader(
    "ここに画像をドラッグ＆ドロップ、または選択（最大5枚まで複数選択可）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="JPEG/PNGに対応。最大5枚まで複数選択可。",
)

# 最大枚数の制限（先頭5枚のみ処理）
MAX_FILES = 5
if uploaded_files and len(uploaded_files) > MAX_FILES:
    st.warning(f"画像は最大{MAX_FILES}枚までです。先頭{MAX_FILES}枚のみ処理します。")
    uploaded_files = uploaded_files[:MAX_FILES]

# 画像が未アップロードでも、デフォルト設定を“無効のまま”見せてガイドする
if not uploaded_files:
    sb.subheader("2. 加工をえらぶ")
    # カテゴリ分けの見本を disabled で提示
    _preview_category_map = {
        "基本": ["なし", "リサイズ", "切り抜き", "回転", "左右反転", "上下反転"],
        "色調": [
            "グレースケール",
            "明るさ調整",
            "コントラスト調整",
            "反転",
            "セピア",
            "ポスタライズ",
            "ソラリゼーション",
        ],
        "効果": ["ぼかし", "シャープ化", "輪郭検出", "エンボス"],
        "モザイク": ["フォトモザイク"],
    }
    sb.radio("カテゴリ", list(_preview_category_map.keys()), index=0, disabled=True)
    sb.selectbox(
        "適用する加工",
        _preview_category_map["基本"],
        index=0,
        disabled=True,
    )

    sb.markdown("#### 出力形式")
    sb.radio(
        "出力形式",
        ["PNG", "JPEG"],
        index=0,
        horizontal=True,
        help="PNG は可逆圧縮で画質劣化なし。JPEG は写真向けでファイルが小さくなります。",
        disabled=True,
    )

    sb.radio(
        "メタデータ（EXIF）の扱い",
        ["保持する", "GPSだけ削除", "全部削除"],
        index=0,
        horizontal=True,
        help=(
            "EXIFは撮影日時やGPSなどのメタデータです。プライバシー配慮が必要な場合は削除を選んでください。"
        ),
        disabled=True,
    )
    st.info("画像をアップロードすると設定を操作できます。")

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)}枚の画像を読み込みました")

    # 2. 加工をえらぶ（サイドバー）
    sb.subheader("2. 加工をえらぶ")

    category_map: dict[str, list[str]] = {
        "基本": ["なし", "リサイズ", "切り抜き", "回転", "左右反転", "上下反転"],
        "色調": [
            "グレースケール",
            "明るさ調整",
            "コントラスト調整",
            "反転",
            "セピア",
            "ポスタライズ",
            "ソラリゼーション",
        ],
        "効果": ["ぼかし", "シャープ化", "輪郭検出", "エンボス"],
        "モザイク": ["フォトモザイク"],
    }
    category = sb.radio("カテゴリ", list(category_map.keys()), index=0)
    process_type = sb.selectbox("適用する加工", category_map[category], index=0)

    # パラメータ設定
    params = {}
    if process_type == "ぼかし":
        params["blur_radius"] = sb.slider("ぼかしの強さ", 0, 10, 2)
    elif process_type == "明るさ調整":
        params["brightness"] = sb.slider("明るさ", 0.5, 2.0, 1.0, 0.1)
    elif process_type == "コントラスト調整":
        params["contrast"] = sb.slider("コントラスト", 0.5, 2.0, 1.0, 0.1)
    elif process_type == "リサイズ":
        sb.markdown("#### リサイズ設定")
        method_label = sb.radio(
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
            params["width"] = sb.number_input("幅 (px)", min_value=1, value=800)
        if params["resize_method"] in ("height", "fit", "stretch"):
            params["height"] = sb.number_input("高さ (px)", min_value=1, value=600)
    elif process_type == "フォトモザイク":
        sb.markdown("#### フォトモザイク設定（タイルセットとサイズを自由に選択）")
        sb.info(
            "⚠️ フォトモザイクは高負荷な処理です。大きな画像は自動でリサイズされます。"
        )
        project_root = Path(__file__).resolve().parent
        tiles_root = project_root / "tiler" / "tiles"

        # tiler/tiles 配下の全タイルセットを列挙（日本語のわかりやすい表示名に変換）
        def friendly_tile_label(fam: str, child: str | None) -> str:
            fam_map = {
                "at": "@",
                "circles": "円",
                "clips": "クリップ",
                "hearts": "ハート",
                "lego": "レゴ",
                "lines": "線",
                "minecraft": "マインクラフト",
                "plus": "＋",
                "times": "×",
                "waves": "波",
            }
            base = fam_map.get(fam, fam)
            if child is None:
                return base
            # 子フォルダごとのニュアンス
            name = child
            if fam == "circles":
                # gen_circle_100 -> 円 100
                import re

                m = re.search(r"(\d+)", name)
                return f"{base} {m.group(1)}" if m else base
            if fam == "lines":
                # gen_line_h/v
                return (
                    f"{base}（横）"
                    if name.endswith("_h")
                    else f"{base}（縦）"
                    if name.endswith("_v")
                    else base
                )
            if fam == "lego":
                return (
                    f"{base}（横）"
                    if name.endswith("_h")
                    else f"{base}（縦）"
                    if name.endswith("_v")
                    else base
                )
            # 単一種別のもの
            if fam in {"plus", "times", "waves", "hearts", "clips", "at"}:
                return base
            return f"{base}/{name}"

        available_sets: list[tuple[str, str]] = []  # (label, abs_path)
        default_index = 0
        if tiles_root.exists():
            families = sorted([p for p in tiles_root.iterdir() if p.is_dir()])
            for fam in families:
                children = sorted([c for c in fam.iterdir() if c.is_dir()])
                if children:
                    for child in children:
                        label = friendly_tile_label(fam.name, child.name)
                        available_sets.append((label, str(child)))
                        # 既定値は circles/gen_circle_100
                        if fam.name == "circles" and child.name == "gen_circle_100":
                            default_index = len(available_sets) - 1
                else:
                    # 直下に画像があるタイプ（例: minecraft）
                    label = friendly_tile_label(fam.name, None)
                    available_sets.append((label, str(fam)))
        else:
            sb.warning(f"タイルルートが見つかりませんでした: {tiles_root}")

        if not available_sets:
            st.error(
                "利用可能なタイルセットが見つかりません。'tiler/tiles' 配下を確認してください。"
            )
            st.stop()

        labels = [lbl for lbl, _ in available_sets]
        tile_label = sb.selectbox(
            "タイルセット", labels, index=min(default_index, len(labels) - 1)
        )
        tile_dir = Path(dict(available_sets)[tile_label])
        if not tile_dir.exists():
            sb.warning(f"タイル画像が見つかりませんでした: {tile_dir}")
        params["tile_dir"] = str(tile_dir)

        # Various sizes（複数サイズ）トグル（先に宣言し、後のノブを無効化制御）
        sb.markdown("##### Various sizes（複数サイズのタイルを混ぜる）")
        use_various = sb.checkbox(
            "複数のタイル倍率を混ぜる",
            value=False,
            help="同じタイルセットから異なる倍率のタイルを複数読み込み、より自然な表現にします（処理時間とメモリ使用量が増えます）。",
        )

        # 主要パラメータ（わかりやすい1つのノブに集約）
        fine_level = sb.slider(
            "目の細かさ（粗い ← 1 … 10 → 超細かい）",
            1,
            10,
            10,
            help="1で高速・粗め、数字が大きいほど細かく（処理が重く・時間がかかり）ます。内部的に画像スケールとタイル倍率を自動調整します。",
            disabled=use_various,
        )
        # レベルごとの推奨プリセット（image_scale, tile_scale）
        presets = [
            (0.5, 1.0),  # 1: 粗い（高速）
            (0.8, 0.8),  # 2
            (1.0, 0.6),  # 3: デフォルト
            (1.2, 0.4),  # 4
            (1.5, 0.25),  # 5
            (1.6, 0.20),  # 6
            (1.8, 0.16),  # 7
            (2.0, 0.14),  # 8
            (2.0, 0.12),  # 9
            (2.0, 0.10),  # 10: 超細かい（非常に重い）
        ]
        image_scale, tile_scale = presets[fine_level - 1]
        sb.caption(f"推奨設定: 画像スケール {image_scale} / タイル倍率 {tile_scale}")

        # 詳細を手動調整したい場合だけ個別スライダーを表示
        with sb.expander("詳細設定", expanded=False):
            custom = sb.checkbox("画像スケール・タイル倍率を手動調整する", value=False)
            if custom:
                image_scale = sb.slider(
                    "画像スケール（大きいほど細かい）",
                    0.2,
                    2.0,
                    image_scale,
                    0.05,
                    help="モザイク元画像の内部解像度。上げるほどタイル数が増えて細かくなります（処理は重くなります）。",
                )
                tile_scale = sb.slider(
                    "タイル倍率（小さいほど細かい）",
                    0.05,
                    1.0,
                    tile_scale,
                    0.01,
                    help="タイル画像自体の拡大縮小倍率。0.5なら半分サイズのタイル＝より細かい目、0.1なら超細かい目になります（非常に重くなります）。",
                )

        # 選択可能な代表倍率一覧
        scale_choices = [1.0, 0.8, 0.6, 0.5, 0.4, 0.33, 0.25, 0.2, 0.16, 0.12, 0.10]

        def nearest_scale(v: float, candidates: list[float]) -> float:
            return min(candidates, key=lambda x: abs(x - v))

        if use_various:
            # 既定で全倍率を選択
            selected_scales = sb.multiselect(
                "使うタイル倍率（値が小さいほど細かい）",
                options=scale_choices,
                default=scale_choices,
                help="複数倍率を混ぜると粒度が混ざってリッチに見えます。",
            )
            if not selected_scales:
                selected_scales = [tile_scale]
        else:
            selected_scales = [tile_scale]

        # 最終的に使う値を params へ格納
        params["image_scale"] = float(image_scale)
        # 詳細（最低限）
        params["color_depth"] = sb.slider(
            "カラー分割（多いほど精密・重くなる）",
            4,
            256,
            64,
            4,
            help="1チャンネルあたりの分割数。64〜128以上は処理・メモリ負荷が高くなります。デフォルト: 64（推奨）",
        )

        # メモリ節約モード
        sb.markdown("##### メモリ節約設定")
        memory_mode = sb.radio(
            "処理モード",
            ["標準（並列処理）", "低メモリ（逐次処理）"],
            help="大きな画像やメモリ不足の場合は「低メモリ」を選択してください。処理は遅くなりますが、安定します。",
        )
        if memory_mode == "低メモリ（逐次処理）":
            auto_pool = 1
        else:
            auto_pool = min(4, max(1, (os.cpu_count() or 4) - 1))  # 8→4に削減
        params["resizing_scales"] = [float(s) for s in selected_scales]
        params["pixel_shift"] = "auto"  # タイルサイズに合わせて固定グリッド
        params["overlap_tiles"] = False
    elif process_type == "切り抜き":
        sb.markdown("#### 切り抜き設定")
        crop_label = sb.radio(
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
            params["size"] = sb.number_input("一辺の長さ (px)", min_value=1, value=512)
        else:
            params["crop_width"] = sb.number_input("幅 (px)", min_value=1, value=512)
            params["crop_height"] = sb.number_input("高さ (px)", min_value=1, value=512)
    elif process_type == "回転":
        params["angle"] = sb.slider("回転角度", 0, 360, 90, 15)
    elif process_type == "ポスタライズ":
        params["bits"] = sb.slider("階調レベル", 1, 8, 2)
    elif process_type == "ソラリゼーション":
        params["threshold"] = sb.slider("しきい値", 0, 255, 128)

    # 出力形式の設定
    sb.markdown("#### 出力形式")
    format_label = sb.radio(
        "出力形式",
        ["PNG", "JPEG"],
        horizontal=True,
        help="PNG は可逆圧縮で画質劣化なし。JPEG は写真向けでファイルが小さくなります。",
    )
    output_format = "PNG" if format_label == "PNG" else "JPEG"
    jpeg_quality = None
    if output_format == "JPEG":
        jpeg_quality = sb.slider(
            "JPEGの品質",
            min_value=60,
            max_value=100,
            value=90,
            help="値が高いほど高画質・大きなファイルになります。",
        )

    # EXIF の扱い
    exif_policy_label = sb.radio(
        "メタデータ（EXIF）の扱い",
        ["保持する", "GPSだけ削除", "全部削除"],
        horizontal=True,
        help=(
            "EXIFは撮影日時やGPSなどのメタデータです。プライバシー配慮が必要な場合は削除を選んでください。"
        ),
    )
    exif_policy_map = {
        "保持する": "keep",
        "GPSだけ削除": "strip_gps",
        "全部削除": "strip_all",
    }
    exif_policy = exif_policy_map[exif_policy_label]

    st.divider()

    # 処理後の画像を保存するリスト（加工なしでもダウンロード可能に）
    processed_images = []

    # 各画像を処理
    # フォトモザイクのタイルは重いので一度だけ読み込んで使い回す
    mosaic_tiles_cache = None
    if process_type == "フォトモザイク":
        try:
            from tiler import build_mosaic_from_pil, load_tiles_with_config
        except Exception:
            st.error(
                "フォトモザイク機能に必要な依存関係が見つかりませんでした。'numpy', 'opencv-python-headless', 'tqdm' をインストールしてください。"
            )
            st.stop()

        @st.cache_resource(show_spinner=True)
        def load_tiles_cached(
            tile_dir: str, color_depth: int, resizing_scales: tuple[float, ...]
        ):
            return load_tiles_with_config(
                [tile_dir], list(resizing_scales), color_depth
            )

        if "tile_dir" in params:
            mosaic_tiles_cache = load_tiles_cached(
                params["tile_dir"],
                int(params.get("color_depth", 32)),
                tuple(params.get("resizing_scales", [1.0])),
            )
    for idx, uploaded_file in enumerate(uploaded_files, 1):
        st.subheader(f"画像 {idx}: {uploaded_file.name}")

        # 画像を読み込み（EXIFの向きを考慮して自動回転）
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)

        # 2カラムレイアウト
        col1, col2 = st.columns(2)

        with col1:
            st.write("**元の画像**")
            st.image(image, use_container_width=True)

        # 画像処理を適用
        if process_type == "フォトモザイク":
            if mosaic_tiles_cache is None:
                st.error("有効なタイルセットを選択してください。")
                st.stop()

            # 高解像度画像の自動リサイズ（メモリ対策）
            MAX_PIXELS = 4_000_000  # 約2000x2000ピクセル相当
            image_pixels = image.width * image.height
            downsample_image = image

            if image_pixels > MAX_PIXELS:
                scale_factor = (MAX_PIXELS / image_pixels) ** 0.5
                new_width = int(image.width * scale_factor)
                new_height = int(image.height * scale_factor)
                downsample_image = image.resize((new_width, new_height), Image.LANCZOS)
                st.warning(
                    f"⚠️ 画像が大きすぎるため、{image.width}x{image.height} → {new_width}x{new_height} にリサイズして処理します。"
                )

            # プログレス表示用
            with st.spinner(
                "フォトモザイクを生成中...（大きな画像は時間がかかります）"
            ):
                # PIL -> mosaic PIL
                processed_image = build_mosaic_from_pil(
                    downsample_image,
                    tiles_paths=None,
                    color_depth=int(params.get("color_depth", 32)),
                    image_scale=float(params.get("image_scale", 0.6)),
                    resizing_scales=list(params.get("resizing_scales", [1.0])),
                    pixel_shift=params.get("pixel_shift", "auto"),
                    pool_size=auto_pool,
                    overlap_tiles=bool(params.get("overlap_tiles", False)),
                    render=False,
                    # use preloaded tiles to avoid reloading per image
                    tiles=mosaic_tiles_cache,
                )
        else:
            processed_image = apply_image_process(image, process_type, params)

        with col2:
            st.write("**加工後の画像**")
            st.image(processed_image, use_container_width=True)

        # 処理後（またはそのまま）の画像を保存
        processed_images.append((processed_image, uploaded_file.name, image))

        # 画像情報表示
        st.caption(
            f"サイズ: {image.size[0]} x {image.size[1]} ピクセル | フォーマット: {image.format} | モード: {image.mode}"
        )

        if idx < len(uploaded_files):
            st.divider()

    # ZIPファイルでまとめてダウンロード
    if len(processed_images) >= 1:
        st.divider()
        st.subheader("📦 3. ダウンロード")

        if len(processed_images) == 1:
            st.info("💡 画像は上の表示エリアから右クリック（長押し）でも保存できます")

        # ZIPファイルを作成
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, (img, original_name, original_img) in enumerate(processed_images, 1):
                # JPEG時のEXIF処理（元画像から取得）
                exif_bytes = None
                if output_format == "JPEG":
                    exif_bytes = build_exif_bytes(original_img, exif_policy)
                img_bytes, ext, _mime = prepare_download_bytes(
                    img, output_format, jpeg_quality, exif_bytes
                )
                # ZIPファイルに追加
                filename = make_download_filename(idx, original_name, ext)
                zip_file.writestr(filename, img_bytes)

        zip_buffer.seek(0)

        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        download_label = (
            f"📥 画像をダウンロード" if len(processed_images) == 1
            else f"🗜️ すべての画像をZIPでダウンロード ({len(processed_images)}枚)"
        )

        st.download_button(
            label=download_label,
            data=zip_buffer.getvalue(),
            file_name=f"processed_images_{timestamp}.zip",
            mime="application/zip",
        )
