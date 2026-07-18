"""Power-Flow 项目的 CPF 连续潮流入口。"""

from pathlib import Path

from src.core.grid import PowerGrid
from src.cpf.parameter import CPFParameter
from src.io.parser import parse_dat_txt
from src.solvers.cpf_solver import ContinuationSolver
from src.utils.config import load_config
from src.visualization.cpf_plotter import CPFPlotter


PROJECT_ROOT = Path(__file__).resolve().parent


def create_cpf_parameters(config):
    """将全局配置转换为 CPF 专用参数。"""
    return CPFParameter(
        lambda_init=config.cpf_lambda_init,
        lambda_min=config.cpf_lambda_min,
        lambda_max=config.cpf_lambda_max,
        step_size=config.cpf_step_size,
        min_step=config.cpf_min_step,
        max_step=config.cpf_max_step,
        step_increase_factor=config.cpf_step_increase_factor,
        step_decrease_factor=config.cpf_step_decrease_factor,
        fast_convergence_iters=config.cpf_fast_convergence_iters,
        slow_convergence_iters=config.cpf_slow_convergence_iters,
        max_step_retries=config.cpf_max_step_retries,
        corrector_tol=config.cpf_corrector_tolerance,
        max_corrector_iters=config.cpf_max_corrector_iterations,
        max_steps=config.cpf_max_steps,
        post_nose_steps=config.cpf_post_nose_steps,
        parameterization=config.cpf_parameterization,
        initial_direction=config.cpf_initial_direction,
        enforce_q_limits=config.cpf_enforce_q_limits,
        tangent_angle_bus=config.cpf_tangent_angle_bus,
        tangent_angle_refinement=config.cpf_tangent_angle_refinement,
        tangent_angle_refinement_cos_threshold=(
            config.cpf_tangent_angle_refinement_cos_threshold
        ),
        tangent_angle_refinement_min_step_ratio=(
            config.cpf_tangent_angle_refinement_min_step_ratio
        ),
        tangent_angle_pseudo_fallback=(
            config.cpf_tangent_angle_pseudo_fallback
        ),
        absolute_vp_angle_bus=config.cpf_absolute_vp_angle_bus,
        absolute_vp_angle_pseudo_fallback=(
            config.cpf_absolute_vp_angle_pseudo_fallback
        ),
    )


def print_cpf_summary(cpf_success, cpf_result, grid):
    """输出 CPF 计算状态和鼻点摘要。"""
    print("\n" + "=" * 64)
    print("                    CPF 连续潮流结果")
    print("-" * 64)
    print(f"计算状态       : {'成功' if cpf_success else '未完成'}")
    print(f"停止原因       : {cpf_result.stop_reason}")
    print(f"参数化方式     : {cpf_result.parameterization}")
    print(
        f"PV→PQ 转换    : "
        f"{'开启' if cpf_result.enforce_q_limits else '关闭'}"
    )
    if cpf_result.tangent_angle_bus is not None:
        print(f"P–V 切线角母线 : {cpf_result.tangent_angle_bus}")
    if cpf_result.absolute_vp_angle_bus is not None:
        print(f"绝对 V/P 角母线: {cpf_result.absolute_vp_angle_bus}")
    print(
        f"伪弧长自动接管 : "
        f"{'是' if cpf_result.used_pseudo_fallback else '否'}"
    )
    print(f"收敛轨迹点数   : {len(cpf_result.points)}")
    print(f"失败尝试次数   : {cpf_result.failed_attempts}")
    print(f"总耗时         : {cpf_result.time_elapsed:.4f} 秒")

    if cpf_result.points:
        nose = cpf_result.nose_point
        base_load_mw = sum(
            float(bus.get("load_mw", 0.0)) for bus in grid.buses
        )
        base_load_mvar = sum(
            float(bus.get("load_mvar", 0.0)) for bus in grid.buses
        )
        print("-" * 64)
        print(f"鼻点所在步数   : {nose.step_index}")
        print(f"鼻点 λ         : {nose.lambda_value:.8f}")
        print(f"鼻点负荷倍率   : {nose.load_multiplier:.8f}")
        print(
            f"鼻点总负荷     : "
            f"{base_load_mw * nose.load_multiplier:.2f} MW / "
            f"{base_load_mvar * nose.load_multiplier:.2f} Mvar"
        )
        print(
            f"鼻点 PQ 最低电压: {nose.min_pq_voltage:.6f} p.u. "
            f"(母线 {nose.min_pq_voltage_bus})"
        )
        print(
            f"鼻点 PQ 最大压降: "
            f"{nose.max_pq_voltage_drop_ratio:.4%} "
            f"(母线 {nose.max_pq_voltage_drop_bus or '无'})"
        )
        limited_buses = nose.q_limited_buses
        print(
            f"鼻点无功越限母线: "
            f"{', '.join(map(str, limited_buses)) if limited_buses else '无'}"
        )
        if nose.pv_tangent_angle_deg is not None:
            print(f"鼻点 P–V 切线角: {nose.pv_tangent_angle_deg:.4f}°")
        if nose.absolute_vp_angle_deg is not None:
            print(f"鼻点绝对 V/P 角: {nose.absolute_vp_angle_deg:.4f}°")
        print(f"鼻点雅可比条件数: {nose.jacobian_condition_number:.6e}")
    print("=" * 64)


def run_cpf(grid, config):
    """运行连续潮流并返回求解结果。"""
    solver = ContinuationSolver(
        params=create_cpf_parameters(config),
        tol=config.cpf_corrector_tolerance,
        max_iter=config.max_iterations,
        verbose=config.cpf_verbose,
    )
    return solver.solve(grid)


def main():
    """读取测试系统、执行 CPF、输出结果并绘制分析曲线。"""
    config = load_config(PROJECT_ROOT / "config.yaml")
    file_path = PROJECT_ROOT / config.data_file

    print("=" * 64)
    print("                  Power-Flow CPF 连续潮流")
    print("=" * 64)

    if not file_path.exists():
        print(f"[错误] 找不到数据文件: {file_path}")
        return

    print(f"\n[1/4] 正在解析测试系统: {file_path}")
    buses, branches = parse_dat_txt(file_path)
    print(f"      已读取 {len(buses)} 条母线、{len(branches)} 条支路")

    print("\n[2/4] 正在建立电网模型 ...")
    grid = PowerGrid(buses, branches, base_mva=config.base_mva)

    if not config.run_cpf:
        print("\n[停止] config.yaml 中的 run_cpf 为 false")
        return

    print("\n[3/4] 正在执行 CPF 预测—校正计算 ...")
    cpf_success, cpf_result = run_cpf(grid, config)
    print_cpf_summary(cpf_success, cpf_result, grid)

    if cpf_result.points:
        print("\n[4/4] 正在绘制 CPF 分析曲线 ...")
        plotter = CPFPlotter(PROJECT_ROOT / config.plot_dir)
        monitor_bus = config.cpf_monitor_bus or None
        output_files = plotter.plot_all(cpf_result, bus_number=monitor_bus)
        print(f">>> 已生成 {len(output_files)} 张 CPF 图像")
        print(f">>> 输出目录: {PROJECT_ROOT / config.plot_dir}")
    else:
        print("\n[4/4] 没有收敛轨迹点，跳过 CPF 绘图")


if __name__ == "__main__":
    main()
