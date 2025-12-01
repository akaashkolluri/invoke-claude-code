"""
Simple Claude Code Invoker — headless, streaming, returns final message + metadata
"""

import os
import sys
import shutil
import json
import time
import subprocess
import shlex
from pathlib import Path
from typing import Optional, Tuple

# -------------------------------------------------------------------
# Allowed tools configuration (safe-ish defaults)
# -------------------------------------------------------------------
ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Bash(python *)",
    "Bash(cat *)",
    "Bash(ls *)",
    "Bash(git *)",
]
# -------------------------------------------------------------------

def _check_cli_exists():
    if not shutil.which("claude"):
        raise RuntimeError("Claude Code CLI not found. Ensure 'claude' is in your PATH.")

def invoke_claude_code(
    directory: str,
    prompt: str,
    stream: bool = True,
    timeout: Optional[int] = 300,
    dangerously_skip_permissions: bool = True,
    model: str | None = "claude-sonnet-4-5",
    limit_tools: bool = True,
    save_logs: bool = True,
) -> Tuple[str, dict]:
    """
    Run Claude Code headlessly and return both:
      - final message (assistant's last text)
      - metadata dict with {time_taken, cost, last_tool_invocation}

    Args:
        directory: working dir
        prompt: user input
        stream: whether to print stream
        timeout: max seconds
        dangerously_skip_permissions: bypass prompts
        model: optional model override
        limit_tools: if True, restrict tool access to ALLOWED_TOOLS; if False, allow everything
        save_logs: whether to save streaming output to scratch/logs.txt

    Returns:
        (final_message, metadata) where metadata includes:
            - time_taken: execution time in seconds
            - cost: API cost if available
            - last_tool_invocation: dict with the last tool that was invoked, including:
                - type: "tool_use" or "tool_result"
                - name: tool name (if tool_use)
                - input: tool input parameters (if tool_use)
                - result: tool result (if available)
    """
    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {dir_path}")

    # Set up logging if requested
    log_file = None
    if save_logs:
        scratch_dir = dir_path / "scratch"
        scratch_dir.mkdir(exist_ok=True)
        log_file = scratch_dir / "logs.txt"
        # Write initial log entry (append mode)
        with open(log_file, 'a') as f:
            f.write(f"\n=== Claude Code Session Started ===\n")
            f.write(f"Directory: {dir_path}\n")
            f.write(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

    _check_cli_exists()

    # Prepend instruction to avoid concurrent tool usage (causes API 400 errors)
    full_prompt = "IMPORTANT: Never use multiple tools in parallel. Always wait for each tool to complete before calling the next one.\n\n" + prompt

    # Build the claude command
    cmd_parts = [
        "claude",
        "-p", full_prompt,
        "--output-format", "stream-json",
        "--verbose",
    ]
    if model:
        cmd_parts += ["--model", model]
    if dangerously_skip_permissions:
        cmd_parts += ["--dangerously-skip-permissions"]
    if limit_tools:
        cmd_parts += ["--allowedTools"] + ALLOWED_TOOLS

    # Properly escape all command parts for shell execution
    escaped_parts = [shlex.quote(part) for part in cmd_parts]
    claude_cmd = " ".join(escaped_parts)
    
    # Ensure PATH includes common locations for node and other tools
    # Get current PATH and add common locations if not present
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep)
    common_paths = ["/usr/local/bin", "/opt/homebrew/bin", "/usr/bin", "/bin"]
    for path in common_paths:
        if path not in path_parts:
            path_parts.insert(0, path)
    enhanced_path = os.pathsep.join(path_parts)
    
    # Wrap command in bash login shell to source ~/.bashrc and ~/.bash_profile
    # This ensures PATH and other environment variables are properly set
    bash_cmd = f"source ~/.bashrc 2>/dev/null || true; source ~/.bash_profile 2>/dev/null || true; {claude_cmd}"
    
    # Use login shell (-l) to ensure proper environment setup
    proc = subprocess.Popen(
        ["bash", "-l", "-c", bash_cmd],
        cwd=str(dir_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "CLICOLOR_FORCE": "1", "PATH": enhanced_path},
    )

    final_message = []
    metadata: dict = {"time_taken": None, "cost": None}
    last_tool_invocation = None  # Track the last tool invocation
    last_result_event = None  # Track the final result event
    start = time.time()

    try:
        assert proc.stdout
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if stream:
                    print(line)
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f"[RAW] {line}\n")
                continue

            etype = event.get("type")
            
            # Debug: log all event types to help diagnose missing messages
            if log_file and etype not in ["message_delta", "content_block_delta"]:
                with open(log_file, 'a') as f:
                    f.write(f"[EVENT_TYPE: {etype}]\n")

            if stream:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            
            # Save all streaming output to log file
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"{line}\n")

            # Capture tool invocations
            if etype == "tool_use":
                # Tool was invoked
                tool_info = {
                    "type": "tool_use",
                    "name": event.get("name"),
                    "input": event.get("input"),
                    "id": event.get("id"),
                }
                last_tool_invocation = tool_info
                
            elif etype == "tool_result":
                # Tool result received
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": event.get("tool_use_id"),
                    "content": event.get("content"),
                    "is_error": event.get("is_error", False),
                }
                # Update last tool invocation with result
                if last_tool_invocation and last_tool_invocation.get("id") == tool_result.get("tool_use_id"):
                    last_tool_invocation["result"] = tool_result
                else:
                    # If no matching invocation, store as standalone result
                    last_tool_invocation = tool_result
                    
            elif etype == "message_delta":
                # Text delta in message
                delta = event.get("delta", {})
                if isinstance(delta, dict):
                    # Check for text directly in delta
                    if "text" in delta:
                        final_message.append(str(delta["text"]))
                    # Also check for content_block_delta
                    if "content_block" in delta:
                        content_block = delta["content_block"]
                        if isinstance(content_block, dict):
                            if "text" in content_block:
                                final_message.append(str(content_block["text"]))
                            elif "type" in content_block and content_block.get("type") == "text" and "text" in content_block:
                                final_message.append(str(content_block["text"]))
                    # Check for nested text in delta
                    if "type" in delta and delta.get("type") == "text" and "text" in delta:
                        final_message.append(str(delta["text"]))
                # Also check top-level text in event (some formats might have it here)
                if "text" in event:
                    final_message.append(str(event["text"]))
                            
            elif etype == "content_block_delta":
                # Content block delta events
                delta = event.get("delta", {})
                if "text" in delta:
                    final_message.append(delta["text"])
                if "index" in event and delta.get("type") == "text":
                    if "text" in delta:
                        final_message.append(delta["text"])
                        
            elif etype == "message":
                # Complete message event (might contain full text)
                if "content" in event:
                    for block in event["content"]:
                        if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                            final_message.append(block["text"])
                            
            elif etype == "message_start":
                # Message start - might have initial content
                if "message" in event:
                    message = event["message"]
                    if "content" in message:
                        for block in message["content"]:
                            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                                final_message.append(block["text"])

            elif etype == "message_stop":
                meta = event.get("metadata", {})
                if "cost" in meta:
                    metadata["cost"] = meta["cost"]
                if "duration" in meta:
                    metadata["time_taken"] = meta["duration"]
                # Check if message is in the event
                if "message" in event:
                    message = event["message"]
                    if "content" in message:
                        for block in message["content"]:
                            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                                final_message.append(block["text"])

            elif etype == "result":
                # Final result event - this is the last message printed to console
                # This should be the primary source of the final message
                last_result_event = event
                # Extract the result text - this is the actual final output
                if "result" in event:
                    result_text = event["result"]
                    # Always use the result as the final message (overwrite any previous)
                    if result_text is not None:
                        final_message = [str(result_text)]
                    else:
                        # If result is None, use the full event as JSON
                        final_message = [json.dumps(event, indent=2)]
                else:
                    # If no result field, use the full event as JSON
                    final_message = [json.dumps(event, indent=2)]
                # Also extract metadata from result event
                if "total_cost_usd" in event:
                    metadata["cost"] = event["total_cost_usd"]
                if "duration_ms" in event:
                    metadata["time_taken"] = event["duration_ms"] / 1000.0  # Convert to seconds
                if "usage" in event:
                    metadata["usage"] = event["usage"]
                if "session_id" in event:
                    metadata["session_id"] = event["session_id"]
                if "num_turns" in event:
                    metadata["num_turns"] = event["num_turns"]
                    
            elif etype == "metadata":
                if "cost" in event:
                    metadata["cost"] = event["cost"]
                if "duration" in event:
                    metadata["time_taken"] = event["duration"]
                    
            # Fallback: try to extract text from any event that might contain it
            # (only if we haven't captured it through the specific handlers above)
            if "text" in event and event["text"]:
                # Only add if not already captured by message_delta
                text = event["text"]
                if isinstance(text, str) and text not in ''.join(final_message):
                    final_message.append(text)
            if "content" in event:
                content = event["content"]
                if isinstance(content, str) and content not in ''.join(final_message):
                    final_message.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                            text = item["text"]
                            if text not in ''.join(final_message):
                                final_message.append(text)
                        elif isinstance(item, str) and item not in ''.join(final_message):
                            final_message.append(item)

        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise TimeoutError("Claude Code timed out.")

    # Read any remaining output after process completes
    if proc.stdout:
        remaining_lines = proc.stdout.readlines()
        for line in remaining_lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                etype = event.get("type")
                
                # Try to extract message from remaining events
                if etype == "message_delta":
                    delta = event.get("delta", {})
                    if "text" in delta:
                        final_message.append(delta["text"])
                elif etype == "message" and "content" in event:
                    for block in event["content"]:
                        if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                            final_message.append(block["text"])
                elif "text" in event:
                    final_message.append(event["text"])
                elif "content" in event:
                    content = event["content"]
                    if isinstance(content, str):
                        final_message.append(content)
            except json.JSONDecodeError:
                # If it's not JSON, might be plain text output
                if line and not line.startswith("[RAW]"):
                    final_message.append(line)

    if proc.returncode != 0:
        raise RuntimeError(f"Claude Code failed with return code {proc.returncode}")

    if metadata["time_taken"] is None:
        metadata["time_taken"] = time.time() - start

    # Write completion log entry
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"=== Claude Code Session Completed ===\n")
            f.write(f"Final message length: {len(''.join(final_message).strip())} chars\n")
            f.write(f"Metadata: {metadata}\n")
            if last_tool_invocation:
                f.write(f"Last tool invocation: {json.dumps(last_tool_invocation, indent=2)}\n")
            f.write(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n")

    # Add last tool invocation to metadata
    if last_tool_invocation:
        metadata["last_tool_invocation"] = last_tool_invocation
    
    # Add last result event to metadata
    if last_result_event:
        metadata["last_result_event"] = last_result_event
        # If we have a result event but no final message, use the result
        if not final_message or not ''.join(final_message).strip():
            if "result" in last_result_event:
                result_text = last_result_event["result"]
                if result_text is not None:
                    final_message = [str(result_text)]

    # Get final message text
    final_message_text = "".join(final_message).strip()
    
    # If still no message and we have a result event, try to extract from it
    if not final_message_text and last_result_event:
        # Try to get result as string representation of the whole event
        final_message_text = json.dumps(last_result_event, indent=2)

    return final_message_text, metadata



## this code shows a test
if __name__ == "__main__":
    msg, meta = invoke_claude_code(
        ".",
        "Write a simple python script that prints 'Hello, world!'",
        stream=True,
        limit_tools=True,   # flip this to False if you want Claude to have free rein
    )
    print("\nFINAL MESSAGE:")
    print(msg)
    print("\nMETADATA:")
    print(meta)
