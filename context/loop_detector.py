from collections import deque
from typing import Any


class LoopDetector:
    """
    loop有两种形式：
        1. repeat: 同一个操作重复多次
        2. cycle: A->B->A->B->... 的循环
    """

    def __init__(self):
        # 完全一样的操作如果重复了如下次数，则认为是循环
        self.max_exact_repeats = 3
        # 如果出现 A->B->A->B->A->B->... 的循环，则认为是循环
        self.max_cycle_length = 3
        self._history: deque = deque(maxlen=20)

    def record_action(self, action_type: str, **details: Any):
        output = [action_type]
        if action_type == "tool_call":
            output.append(details.get("tool_name", ""))
            args = details.get("args", {})
            if isinstance(args, dict):
                for k in sorted(args.keys()):
                    output.append(f"{k}={str(args[k])}")
        elif action_type == "response":  # AI text response
            output.append(details.get("text", ""))
        signature = "|".join(output)
        self._history.append(signature)
        return signature

    def check_for_loop(self) -> str | None:
        if len(self._history) < 2:
            return None

        if len(self._history) >= self.max_exact_repeats:
            # 最后 max_exact_repeats次操作是否完全相同
            recent = list(self._history)[-self.max_exact_repeats :]
            if len(set(recent)) == 1:
                return f"Same action repeated {self.max_exact_repeats}"

            if len(self._history) >= self.max_cycle_length * 2:
                history = list(self._history)
                for cycle_len in range(
                    2, min(self.max_cycle_length + 1, len(history) // 2 + 1)
                ):
                    recent = history[-cycle_len * 2 :]
                    if recent[:cycle_len] == recent[cycle_len:]:
                        return f"Cycle detected with length {cycle_len}"
