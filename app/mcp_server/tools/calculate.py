"""Calculate tools for MACP agents.

Tools for basic calculations and math operations.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Dict, List


def register_calculate_tools(mcp_server) -> None:
    """Register calculate tools to the MCP server."""

    if hasattr(mcp_server, "tool"):
        @mcp_server.tool()
        def calculate_math(expression: str) -> str:
            """Evaluate a mathematical expression.

            Args:
                expression: Math expression to evaluate (e.g., "2 + 3 * 4").

            Returns:
                JSON string with result.
            """
            try:
                # Safe evaluation with only math operations
                allowed_names = {
                    "abs": abs, "round": round, "min": min, "max": max,
                    "sum": sum, "len": len,
                    "sqrt": math.sqrt, "pow": math.pow,
                    "sin": math.sin, "cos": math.cos, "tan": math.tan,
                    "log": math.log, "log10": math.log10, "exp": math.exp,
                    "pi": math.pi, "e": math.e,
                    "ceil": math.ceil, "floor": math.floor,
                }
                # Replace common notation
                expr = expression.replace("^", "**")
                result = eval(expr, {"__builtins__": {}}, allowed_names)
                return json.dumps({
                    "expression": expression,
                    "result": result,
                    "status": "ok",
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "expression": expression,
                    "error": str(e),
                    "status": "error",
                })

        @mcp_server.tool()
        def calculate_stats(numbers: str) -> str:
            """Calculate statistics for a list of numbers.

            Args:
                numbers: JSON string or comma-separated list of numbers.

            Returns:
                JSON string with statistics (mean, median, min, max, etc).
            """
            try:
                if numbers.startswith("["):
                    nums = json.loads(numbers)
                else:
                    nums = [float(x.strip()) for x in numbers.split(",") if x.strip()]
                
                if not nums:
                    return json.dumps({
                        "error": "No numbers provided",
                        "status": "error",
                    })
                
                nums = [float(x) for x in nums]
                sorted_nums = sorted(nums)
                n = len(nums)
                mean = sum(nums) / n
                median = sorted_nums[n // 2] if n % 2 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
                
                return json.dumps({
                    "count": n,
                    "sum": sum(nums),
                    "mean": round(mean, 4),
                    "median": round(median, 4),
                    "min": min(nums),
                    "max": max(nums),
                    "range": max(nums) - min(nums),
                    "variance": round(sum((x - mean) ** 2 for x in nums) / n, 4),
                    "std_dev": round(math.sqrt(sum((x - mean) ** 2 for x in nums) / n), 4),
                    "status": "ok",
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "numbers": numbers,
                    "error": str(e),
                    "status": "error",
                })

        @mcp_server.tool()
        def calculate_format_number(value: float, decimals: int = 2) -> str:
            """Format a number with specified decimal places.

            Args:
                value: Number to format.
                decimals: Number of decimal places (default: 2).

            Returns:
                JSON string with formatted number.
            """
            try:
                formatted = round(float(value), decimals)
                return json.dumps({
                    "original": value,
                    "formatted": formatted,
                    "decimals": decimals,
                    "status": "ok",
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "value": value,
                    "error": str(e),
                    "status": "error",
                })

        @mcp_server.tool()
        def calculate_time_diff(start_time: str, end_time: str) -> str:
            """Calculate time difference between two timestamps.

            Args:
                start_time: Start timestamp (ISO format or epoch seconds).
                end_time: End timestamp (ISO format or epoch seconds).

            Returns:
                JSON string with time difference in various units.
            """
            try:
                # Parse timestamps
                def parse_time(ts: str):
                    try:
                        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except ValueError:
                        return datetime.fromtimestamp(float(ts))
                
                start = parse_time(start_time)
                end = parse_time(end_time)
                diff = end - start
                diff_seconds = diff.total_seconds()
                
                return json.dumps({
                    "start_time": str(start),
                    "end_time": str(end),
                    "diff_seconds": round(diff_seconds, 2),
                    "diff_minutes": round(diff_seconds / 60, 2),
                    "diff_hours": round(diff_seconds / 3600, 2),
                    "diff_days": round(diff_seconds / 86400, 2),
                    "status": "ok",
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "start_time": start_time,
                    "end_time": end_time,
                    "error": str(e),
                    "status": "error",
                })

    elif hasattr(mcp_server, "add_tool"):
        mcp_server.add_tool(calculate_math)
        mcp_server.add_tool(calculate_stats)
        mcp_server.add_tool(calculate_format_number)
        mcp_server.add_tool(calculate_time_diff)


# Standalone functions for direct import
def calculate_math(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len,
            "sqrt": math.sqrt, "pow": math.pow,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
        }
        expr = expression.replace("^", "**")
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return json.dumps({
            "expression": expression,
            "result": result,
            "status": "ok",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "expression": expression,
            "error": str(e),
            "status": "error",
        })


def calculate_stats(numbers: str) -> str:
    """Calculate statistics for a list of numbers."""
    try:
        if numbers.startswith("["):
            nums = json.loads(numbers)
        else:
            nums = [float(x.strip()) for x in numbers.split(",") if x.strip()]
        
        if not nums:
            return json.dumps({
                "error": "No numbers provided",
                "status": "error",
            })
        
        nums = [float(x) for x in nums]
        sorted_nums = sorted(nums)
        n = len(nums)
        mean = sum(nums) / n
        median = sorted_nums[n // 2] if n % 2 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        
        return json.dumps({
            "count": n,
            "sum": sum(nums),
            "mean": round(mean, 4),
            "median": round(median, 4),
            "min": min(nums),
            "max": max(nums),
            "range": max(nums) - min(nums),
            "variance": round(sum((x - mean) ** 2 for x in nums) / n, 4),
            "std_dev": round(math.sqrt(sum((x - mean) ** 2 for x in nums) / n), 4),
            "status": "ok",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "numbers": numbers,
            "error": str(e),
            "status": "error",
        })


def calculate_format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places."""
    try:
        formatted = round(float(value), decimals)
        return json.dumps({
            "original": value,
            "formatted": formatted,
            "decimals": decimals,
            "status": "ok",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "value": value,
            "error": str(e),
            "status": "error",
        })


def calculate_time_diff(start_time: str, end_time: str) -> str:
    """Calculate time difference between two timestamps."""
    try:
        def parse_time(ts: str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return datetime.fromtimestamp(float(ts))
        
        start = parse_time(start_time)
        end = parse_time(end_time)
        diff = end - start
        diff_seconds = diff.total_seconds()
        
        return json.dumps({
            "start_time": str(start),
            "end_time": str(end),
            "diff_seconds": round(diff_seconds, 2),
            "diff_minutes": round(diff_seconds / 60, 2),
            "diff_hours": round(diff_seconds / 3600, 2),
            "diff_days": round(diff_seconds / 86400, 2),
            "status": "ok",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "start_time": start_time,
            "end_time": end_time,
            "error": str(e),
            "status": "error",
        })