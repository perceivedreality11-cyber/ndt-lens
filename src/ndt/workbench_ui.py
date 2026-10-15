"""Local image annotation, training, model selection, and batch prediction UI."""

import base64
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ndt.detection import CLASSES, load_model
from ndt.export import export_workspace
from ndt.imaging import load_image
from ndt.jobs import list_jobs, start_training
from ndt.workspace import Workspace

EDITOR = components.declare_component("ndt_box_editor", path=str(Path(__file__).parent / "box_editor"))
CLASS_NAMES = {"porosity": "气孔 · porosity", "crack": "裂纹 · crack", "inclusion": "夹杂 · inclusion"}
STATUS = {"queued": "排队中", "running": "训练中", "completed": "训练完成", "failed": "失败"}


@st.cache_resource
def cached_model(path, modified, size):
    return load_model(path)


def selected_model(path):
    file = Path(path)
    if not file.is_file():
        raise ValueError("请先完成训练或指定可用的本地权重。")
    info = file.stat()
    return cached_model(str(file.resolve()), info.st_mtime_ns, info.st_size)


def model_selector(workspace):
    jobs = list_jobs(workspace.root)
    models = {
        j["id"]: j["weights"] for j in jobs if j["status"] == "completed" and Path(j["weights"]).is_file()
    }
    choices = ["暂不使用模型", *models, "指定本地权重"]
    selected = st.sidebar.selectbox("自动标注模型", choices)
    if selected == "指定本地权重":
        return st.sidebar.text_input("可信的缺陷模型 .pt 路径")
    return models.get(selected, "")


def open_demo():
    from ndt.demo import create_demo_workspace

    workspace = create_demo_workspace("data/demo-workbench")
    st.session_state["workspace_root"] = str(workspace.root)


def render(page):
    st.caption("ANNOTATE → TRAIN → AUTO LABEL → REVIEW")
    st.title(
        {"图片与标注": "图片与标注", "训练中心": "让模型学习你的标注", "批量自动标注": "自动定位，逐张复核"}[
            page
        ]
    )
    notice = st.session_state.pop("workbench_notice", None)
    if notice:
        st.success(notice)
    st.sidebar.subheader("本地工作空间")
    root = st.sidebar.text_input("数据保存目录", "data/workbench", key="workspace_root")
    st.sidebar.button("打开合成演示项目", on_click=open_demo)
    st.sidebar.caption("演示无需上传文件或下载模型；所有图片均为程序合成。")
    try:
        workspace = Workspace(root)
        if workspace.metadata().get("synthetic"):
            st.warning("合成演示项目：已确认标注为程序生成的真值，用于体验流程，不能说明真实检测性能。")
        weights = model_selector(workspace)
        confidence = st.sidebar.slider("自动标注置信度", 0.05, 0.95, 0.25)
        rows = workspace.images()
        a, b, c, d = st.columns(4)
        a.metric("图片总数", len(rows))
        b.metric("已确认标注", sum(r["reviewed"] for r in rows))
        c.metric("模型标注待复核", sum(bool(r["model"]) and not r["reviewed"] for r in rows))
        d.metric("已确认缺陷框", sum(len(json.loads(r["boxes"])) for r in rows if r["reviewed"]))
        with st.expander("数据集就绪情况", expanded=not rows):
            st.dataframe(
                [
                    {
                        "用途": split,
                        "图片": sum(r["split"] == split for r in rows),
                        "已确认": sum(r["split"] == split and bool(r["reviewed"]) for r in rows),
                        "工件组": len({r["group_id"] for r in rows if r["split"] == split}),
                    }
                    for split in ("train", "val", "test")
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "train 学习参数 · val 选择模型 · test 最终评估。至少确认来自不同工件组的 train 和 val 后才能训练。"
            )
        if page == "图片与标注":
            annotation_page(workspace, weights, confidence)
        elif page == "训练中心":
            training_page(workspace, weights)
        else:
            batch_page(workspace, weights, confidence)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        st.error(str(exc))


def annotation_page(workspace, weights, confidence):
    with st.expander("导入一批射线图片", expanded=not workspace.images()):
        st.caption(
            "按工件/底片分组导入，同一组只能属于一个数据集。原始文件单独保留，标注基于统一转换后的 PNG。"
        )
        uploads = st.file_uploader(
            "选择图片（支持多选）",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
        )
        group = st.text_input("工件 / 底片组号", placeholder="例如：WELD-2026-001")
        split = st.selectbox(
            "数据集用途",
            ["train", "val", "test"],
            format_func=lambda x: {
                "train": "train · 学习用",
                "val": "val · 验证与选模型",
                "test": "test · 最终独立评估",
            }[x],
        )
        automatic = st.checkbox(
            "导入后立即用所选模型自动标注", value=bool(weights), disabled=not bool(weights)
        )
        if st.button("导入图片", type="primary", disabled=not uploads):
            model = selected_model(weights) if automatic else None
            imported, duplicate, errors = [], 0, []
            for upload in uploads:
                try:
                    image_id, created = workspace.import_image(upload.name, upload.getvalue(), group, split)
                    if created:
                        imported.append(image_id)
                    else:
                        duplicate += 1
                except (ValueError, OSError) as exc:
                    errors.append(f"{upload.name}: {exc}")
            st.success(f"新增 {len(imported)} 张；跳过重复图片 {duplicate} 张。")
            for error in errors:
                st.error(error)
            if model is not None and imported:
                bar = st.progress(0.0, text="自动标注新图片…")
                count = workspace.auto_label(imported, model, weights, confidence, bar.progress)
                st.success(f"自动标注完成：{count} 张，等待人工复核。")
    rows = workspace.images()
    if not rows:
        st.info("先导入一批图片，画框并确认，再到训练中心启动学习。")
        return
    filter_status = st.radio("查看范围", ["全部图片", "待复核", "已确认"], horizontal=True)
    filtered = [
        r for r in rows if filter_status == "全部图片" or bool(r["reviewed"]) == (filter_status == "已确认")
    ]
    if not filtered:
        st.info("当前范围没有图片。")
        return
    by_id = {r["id"]: r for r in filtered}
    preferred = next((i for i, r in enumerate(filtered) if json.loads(r["boxes"])), 0)
    image_id = st.selectbox(
        "选择图片",
        list(by_id),
        index=preferred,
        format_func=lambda i: (
            f"{'✓' if by_id[i]['reviewed'] else '○'} {by_id[i]['name']}  /  {by_id[i]['group_id']}  /  {by_id[i]['split']}"
        ),
    )
    row = workspace.get(image_id)
    st.caption(f"{row['width']} × {row['height']} px  ·  {'已确认' if row['reviewed'] else '待复核'}")
    if row["model"]:
        st.caption(f"自动标注来源：{row['model']}")
    st.caption("在图像上拖动添加框；表格可修改类别、坐标或删除行。完成后点击保存。坐标为原图像素。")
    label = st.selectbox("绘制类别", CLASSES, format_func=CLASS_NAMES.get)
    zoom = st.select_slider("画布缩放（放大后可滚动）", options=[1, 1.5, 2, 3], value=1)
    image = load_image(workspace.image_path(image_id))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    key = f"draft_{image_id}_{row['revision']}"
    if key not in st.session_state:
        st.session_state[key] = json.loads(row["boxes"])
    result = EDITOR(
        image="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
        image_id=image_id,
        revision=row["revision"],
        boxes=st.session_state[key],
        label=label,
        zoom=zoom,
        key=f"canvas_{image_id}_{row['revision']}",
        default=None,
    )
    event_key = f"event_{key}"
    if result and result.get("image_id") == image_id and result.get("revision") == row["revision"]:
        if result.get("seq") != st.session_state.get(event_key):
            st.session_state[key] = result["boxes"]
            st.session_state[event_key] = result["seq"]
    boxes = st.session_state[key]
    frame = pd.DataFrame(
        [
            dict(label=b["label"], x1=b["xyxy"][0], y1=b["xyxy"][1], x2=b["xyxy"][2], y2=b["xyxy"][3])
            for b in boxes
        ],
        columns=["label", "x1", "y1", "x2", "y2"],
    )
    # A new canvas event rebuilds the table; table edits remain stable on other reruns.
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"table_{key}_{st.session_state.get(event_key, 0)}",
        column_config={"label": st.column_config.SelectboxColumn("缺陷类别", options=CLASSES, required=True)},
    )
    st.caption("删除所有框后确认，可将该图记录为已复核的正常样本。空预测也需要人工确认。")
    a, b = st.columns(2)
    confirm = a.button("保存并确认标注", type="primary")
    draft = b.button("保存草稿（不参与训练）")
    if confirm or draft:
        edited_boxes = [
            dict(label=r["label"], xyxy=[r["x1"], r["y1"], r["x2"], r["y2"]])
            for r in edited.to_dict("records")
        ]
        workspace.save_boxes(
            image_id, edited_boxes, reviewed=confirm, model=row["model"], expected_revision=row["revision"]
        )
        st.rerun()


def training_page(workspace, weights):
    st.write("已确认的 train 图片用于学习，val 用于验证。训练在后台运行，切换页面后仍会继续。")
    st.info("建议先标注多批真实工件，并分别导入 train / val。确认后的自动标注可用于下一轮训练。")
    synthetic = workspace.metadata().get("synthetic", False)
    with st.form("training"):
        a, b, c = st.columns(3)
        epochs = a.number_input("训练轮数", 1, 1000, 1 if synthetic else 100)
        imgsz = b.selectbox("训练图像边长", [640, 1024, 1280, 128, 256], index=3 if synthetic else 0)
        batch = c.number_input("批次大小", 1, 128, 2 if synthetic else 8)
        device = st.selectbox(
            "计算设备", ["cpu", "0"], format_func=lambda x: "CPU" if x == "cpu" else "CUDA GPU 0"
        )
        init = st.selectbox(
            "初始化方式",
            ["通用预训练权重", "从零训练（无需下载）", "继续学习所选缺陷模型"],
            index=1 if synthetic else 0,
        )
        st.caption(
            "通用预训练权重首次需要联网下载；继续学习会创建新模型。CPU 较慢，可先用 1 轮、128 边长验证流程。"
        )
        submit = st.form_submit_button("从已确认标注启动训练", type="primary")
    if submit:
        model = "yolo11n.pt" if init == "通用预训练权重" else "yolo11n.yaml"
        if init == "继续学习所选缺陷模型":
            selected_model(weights)
            model = str(Path(weights).resolve())
        path = workspace.snapshot()
        job_path = start_training(
            workspace.root, path, epochs=epochs, imgsz=imgsz, batch=batch, device=device, model=model
        )
        st.success(f"训练已启动：{job_path.parent.name}")
    st.button("刷新训练状态")
    jobs = list_jobs(workspace.root)
    if not jobs:
        st.caption("还没有训练记录。")
    for job in jobs:
        with st.expander(f"{STATUS[job['status']]}  ·  {job['id']}", expanded=job["status"] != "completed"):
            st.write(f"{job['epochs']} 轮 · {job['imgsz']} px · {job['device']}")
            if job.get("error"):
                st.error(job["error"])
            if job.get("weights"):
                st.success("训练成功。请在侧栏选择此模型，即可自动标注后续图片。")
                st.code(job["weights"], language=None)
            folder = Path(job["directory"])
            curves = list((folder / "result").glob("*/results.csv"))
            if curves:
                try:
                    data = pd.read_csv(curves[0])
                    data.columns = data.columns.str.strip()
                    columns = [c for c in ("metrics/mAP50(B)", "metrics/mAP50-95(B)") if c in data]
                    if columns:
                        st.line_chart(data.set_index("epoch")[columns])
                except (OSError, ValueError):
                    st.caption("训练结果正在写入，请稍后刷新。")
            log = folder / "train.log"
            if log.is_file():
                st.code(log.read_text(encoding="utf-8", errors="replace")[-8000:], language=None)


def batch_page(workspace, weights, confidence):
    rows = workspace.images()
    pending = [r for r in rows if not r["reviewed"]]
    st.write(f"待复核图片 {len(pending)} 张。自动标注后，到“图片与标注”逐张修正并确认。")
    st.caption("会重新生成待复核图片的标注，包括未确认草稿；已人工确认的图片会保留。")
    if not weights:
        st.info("请先在训练中心训练模型，并在侧栏选择它；也可以指定已有的缺陷模型。")
    if st.button("自动标注所有待复核图片", type="primary", disabled=not weights or not pending):
        model = selected_model(weights)
        bar = st.progress(0.0, text="正在批量标注…")
        count = workspace.auto_label([r["id"] for r in pending], model, weights, confidence, bar.progress)
        st.session_state["workbench_notice"] = f"已完成 {count} 张图片的自动标注，请进行人工复核。"
        st.rerun()
    st.divider()
    reviewed_only = st.checkbox("仅导出已确认标注（适合训练）", value=True)
    signature = (
        str(workspace.root),
        reviewed_only,
        tuple((r["id"], r["revision"]) for r in workspace.images()),
    )
    if st.button("生成标注结果下载包", disabled=not rows):
        st.session_state["annotation_export"] = (
            signature,
            export_workspace(workspace, reviewed_only=reviewed_only),
        )
    saved = st.session_state.get("annotation_export")
    if saved and saved[0] == signature:
        st.download_button(
            "下载 ZIP（原图 / 标注图 / YOLO 数据集 / 复核记录）",
            saved[1],
            "ndt-annotations.zip",
            "application/zip",
        )
