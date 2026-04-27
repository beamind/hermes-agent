#!/usr/bin/env bash
# Voice Gateway 一键启动/停止/状态脚本
#
# Usage:
#   bash run.sh start   # 后台启动 gateway
#   bash run.sh stop    # 停止 gateway
#   bash run.sh status  # 查看运行状态
#   bash run.sh restart # 重启 gateway
#   bash run.sh logs    # 查看实时日志

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="hermes-gateway"
PID_FILE="${SCRIPT_DIR}/${PROJECT_NAME}.pid"
LOG_FILE="${SCRIPT_DIR}/${PROJECT_NAME}.log"

# ------------------------------------------------------------------
# 查找 Python
# ------------------------------------------------------------------
PYTHON=""
for candidate in \
    "${SCRIPT_DIR}/venv/bin/python" \
    "${SCRIPT_DIR}/.venv/bin/python" \
    "${HOME}/.hermes/hermes-agent/venv/bin/python" \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"
do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
        PYTHON="${candidate}"
        break
    fi
done

if [[ -z "${PYTHON}" ]]; then
    echo "❌ 找不到 Python 解释器，请确认 venv 已创建" >&2
    exit 1
fi

# ------------------------------------------------------------------
# start
# ------------------------------------------------------------------
cmd_start() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo "✅ gateway 已在运行 (PID: ${pid})"
            return 0
        fi
        rm -f "${PID_FILE}"
    fi

    echo "🚀 启动 Hermes Gateway (voice)..."
    echo "   Python: ${PYTHON}"
    echo "   Log:    ${LOG_FILE}"

    cd "${SCRIPT_DIR}"
    nohup "${PYTHON}" -m gateway.run >> "${LOG_FILE}" 2>&1 &
    local pid=$!
    echo "${pid}" > "${PID_FILE}"

    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
        echo "✅ 启动成功 (PID: ${pid})"
        echo "   查看日志: bash run.sh logs"
    else
        echo "❌ 启动失败，请查看日志: ${LOG_FILE}"
        rm -f "${PID_FILE}"
        return 1
    fi
}

# ------------------------------------------------------------------
# stop
# ------------------------------------------------------------------
cmd_stop() {
    if [[ ! -f "${PID_FILE}" ]]; then
        echo "ℹ️ gateway 未运行 (无 PID 文件)"
        return 0
    fi

    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -z "${pid}" ]]; then
        echo "ℹ️ PID 文件为空，清理中..."
        rm -f "${PID_FILE}"
        return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "ℹ️ 进程 ${pid} 已不存在，清理 PID 文件..."
        rm -f "${PID_FILE}"
        return 0
    fi

    echo "🛑 停止 gateway (PID: ${pid})..."
    kill "${pid}" 2>/dev/null || true

    local waited=0
    while kill -0 "${pid}" 2>/dev/null && (( waited < 10 )); do
        sleep 1
        ((waited++)) || true
    done

    if kill -0 "${pid}" 2>/dev/null; then
        echo "   强制终止..."
        kill -9 "${pid}" 2>/dev/null || true
    fi

    rm -f "${PID_FILE}"
    echo "✅ 已停止"
}

# ------------------------------------------------------------------
# status
# ------------------------------------------------------------------
cmd_status() {
    if [[ ! -f "${PID_FILE}" ]]; then
        echo "🔴 gateway 未运行"
        return 0
    fi

    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -z "${pid}" ]]; then
        echo "🟡 PID 文件异常 (空)"
        return 0
    fi

    if kill -0 "${pid}" 2>/dev/null; then
        echo "🟢 gateway 运行中 (PID: ${pid})"
        echo "   Python: ${PYTHON}"
        echo "   Log:    ${LOG_FILE}"
        echo ""
        echo "   最近 5 条日志:"
        tail -n 5 "${LOG_FILE}" 2>/dev/null || echo "   (日志文件为空或不存在)"
    else
        echo "🔴 gateway 未运行 (PID ${pid} 已失效)"
        rm -f "${PID_FILE}"
    fi
}

# ------------------------------------------------------------------
# restart
# ------------------------------------------------------------------
cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start
}

# ------------------------------------------------------------------
# logs
# ------------------------------------------------------------------
cmd_logs() {
    if [[ ! -f "${LOG_FILE}" ]]; then
        echo "ℹ️ 日志文件不存在: ${LOG_FILE}"
        return 0
    fi
    echo "📜 实时日志 (Ctrl+C 退出)..."
    tail -n 50 -f "${LOG_FILE}"
}

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
case "${1:-}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    restart)
        cmd_restart
        ;;
    logs)
        cmd_logs
        ;;
    *)
        echo "Usage: bash run.sh {start|stop|status|restart|logs}"
        exit 1
        ;;
esac
