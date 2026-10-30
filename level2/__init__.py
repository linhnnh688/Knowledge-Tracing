"""Cấp độ 2 — tái lập GRKT (KDD 2024): mô tả năng lực người học trên đồ thị khái niệm.

Quy ước: ``GRKT/`` là mã nguồn mở đã chỉnh sửa (import tương đối + hook ghi quỹ đạo
mastery); mọi logic thực nghiệm nằm ở package này, notebook chỉ điều phối — giống
cách ``level1/`` phục vụ ``kaggle_level1.ipynb``.

Điểm bắt buộc của Cấp độ 2 (YEU_CAU.md §5, §6, §7):

* dùng LẠI đúng split theo người học của Cấp độ 1 (seed 42) cho mọi mô hình;
* đồ thị khái niệm (``rel_map`` / ``pre_map`` của GRKT) chỉ dựng từ tập train;
* đánh giá trên ≥ 2 tập dữ liệu công khai với cùng quy trình chia và cùng chỉ số;
* trực quan hóa mastery theo khái niệm × thời gian và đo ≥ 2 tiêu chí hợp lý của quỹ đạo.
"""

from .dataset_spec import (  # noqa: F401
    ASSIST2009,
    ASSIST2009_SMOKE,
    CANONICAL_COLUMNS,
    JUNYI,
    DatasetSpec,
    ensure_canonical,
    find_kaggle_input,
    get_spec,
    probe_columns,
    register_spec,
    resolve_canonical_path,
)
from .grkt_data import (  # noqa: F401
    GRKTData,
    UserSequence,
    adjacency_from_graphs,
    build_grkt_data,
    build_relevance_maps,
    graph_variant,
    leak_report,
    resolve_time_mode,
    split_sequences,
)
from .grkt_api import (  # noqa: F401
    build_grkt,
    ensure_grkt_importable,
    evaluate_grkt,
    import_grkt,
    make_args,
    predict_grkt,
    run_grkt_experiment,
    train_grkt,
)
from .trajectory import (  # noqa: F401
    directional_consistency,
    export_mastery_csv,
    forgetting_monotonicity,
    load_trajectories,
    locality,
    plot_mastery_curves,
    plot_mastery_heatmap,
    record_trajectories,
    save_trajectories,
    stage_attribution,
    trajectory_metrics_table,
)
from .calibration import (  # noqa: F401
    brier_score,
    calibration_table,
    expected_calibration_error,
    reliability_plot,
    save_reliability_rows,
)
from .baselines import collect_level1_probs, level1_baseline  # noqa: F401
from .experiment import run_dataset, run_level2  # noqa: F401
