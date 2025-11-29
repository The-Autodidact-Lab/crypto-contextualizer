#!/usr/bin/env python3
"""
Script to run all ss, ms, sm, and mm scenarios with multi_agent configuration
and parse the outputs.

Usage:
    python run_all_scenarios.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

# Configuration
MODEL = "openai/gpt-5-mini"
PROVIDER = "llama-api"
AGENT = "multi_agent"

# Scenario prefixes to discover
SCENARIO_PREFIXES = ["ss", "ms", "sm", "mm"]


def discover_scenarios() -> List[str]:
    """
    Discover all registered scenarios matching the prefixes (ss*, ms*, sm*, mm*).
    
    Returns:
        List of scenario IDs
    """
    try:
        # Use the ARE CLI to list all scenarios
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "are.simulation.main",
                "--list-scenarios",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        
        all_scenarios = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        
        # Filter scenarios by prefix
        matching_scenarios = []
        for scenario_id in all_scenarios:
            for prefix in SCENARIO_PREFIXES:
                if scenario_id.startswith(prefix):
                    matching_scenarios.append(scenario_id)
                    break
        
        return sorted(matching_scenarios)
    except subprocess.CalledProcessError as e:
        print(f"Error discovering scenarios: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error discovering scenarios: {e}")
        sys.exit(1)


def run_scenarios(scenario_ids: List[str], output_dir: str) -> str:
    """
    Run all scenarios with the specified configuration.
    
    Args:
        scenario_ids: List of scenario IDs to run
        output_dir: Directory to save output files
        
    Returns:
        Path to the output JSONL file
    """
    print(f"\n{'='*80}")
    print(f"Running {len(scenario_ids)} scenarios")
    print(f"Model: {MODEL}")
    print(f"Provider: {PROVIDER}")
    print(f"Agent: {AGENT}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        sys.executable,
        "-m",
        "are.simulation.main",
        "-m", MODEL,
        "--provider", PROVIDER,
        "-a", AGENT,
        "--export",
        "--output_dir", output_dir,
    ]
    
    # Add all scenario IDs
    for scenario_id in scenario_ids:
        cmd.extend(["-s", scenario_id])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output stream to console
        )
        print("\n✓ Scenarios completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running scenarios: {e}")
        sys.exit(1)
    
    # Check for output file
    output_file = os.path.join(output_dir, "output.jsonl")
    if not os.path.exists(output_file):
        print(f"Warning: Expected output file '{output_file}' not found")
        return ""
    
    return output_file


def parse_results(result_file: str) -> Dict[str, Dict]:
    """
    Parse the output JSONL file and extract results.
    
    Args:
        result_file: Path to the output JSONL file
        
    Returns:
        Dictionary mapping scenario IDs to their results
    """
    results = {}
    
    if not os.path.exists(result_file):
        print(f"Error: Results file '{result_file}' not found")
        return results
    
    with open(result_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                rec = json.loads(line)
                
                # Extract metadata
                md = rec.get("metadata", {}) or {}
                scenario_id = md.get("scenario_id") or rec.get("task_id") or f"unknown_{line_num}"
                
                # Extract status information
                status = md.get("status")
                has_exc = md.get("has_exception", False)
                success = (status == "success") and not has_exc
                
                exception_type = md.get("exception_type") or ""
                exception_message = md.get("exception_message") or ""
                rationale = md.get("rationale") or ""
                duration = md.get("run_duration") or md.get("duration")
                
                results[scenario_id] = {
                    "scenario_id": scenario_id,
                    "success": success,
                    "status": status,
                    "has_exception": has_exc,
                    "exception_type": exception_type,
                    "exception_message": exception_message,
                    "rationale": rationale,
                    "duration": duration,
                }
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    return results


def print_summary(results: Dict[str, Dict], scenario_ids: List[str]):
    """
    Print a formatted summary of the results.
    
    Args:
        results: Dictionary of parsed results
        scenario_ids: List of all scenario IDs that were run
    """
    print(f"\n{'='*80}")
    print("SCENARIO VALIDATION RESULT SUMMARY")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Scenario ID':<15} {'Success':<10} {'Status':<12} {'Exception':<50}")
    print("-" * 90)
    
    # Print results for each scenario
    success_count = 0
    failed_count = 0
    exception_count = 0
    no_result_count = 0
    
    for scenario_id in scenario_ids:
        if scenario_id in results:
            result = results[scenario_id]
            success = result["success"]
            status = result["status"] or "unknown"
            has_exc = result["has_exception"]
            exc_msg = result["exception_message"] or result["exception_type"] or ""
            
            if success:
                success_count += 1
            elif has_exc:
                exception_count += 1
            else:
                failed_count += 1
            
            success_str = "✓ PASS" if success else "✗ FAIL"
            exc_display = exc_msg[:47] if exc_msg else ""
            
            print(f"{scenario_id:<15} {success_str:<10} {status:<12} {exc_display:<50}")
        else:
            no_result_count += 1
            print(f"{scenario_id:<15} {'NO RESULT':<10} {'missing':<12} {'':<50}")
    
    # Summary statistics
    print(f"\n{'-'*90}")
    print(f"Total scenarios: {len(scenario_ids)}")
    print(f"  ✓ Successful: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  ⚠ Exceptions: {exception_count}")
    print(f"  ? No result: {no_result_count}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    print("Discovering scenarios...")
    scenario_ids = discover_scenarios()
    
    if not scenario_ids:
        print("No matching scenarios found!")
        print(f"Looking for scenarios with prefixes: {', '.join(SCENARIO_PREFIXES)}")
        sys.exit(1)
    
    print(f"Found {len(scenario_ids)} scenarios: {', '.join(scenario_ids)}")
    
    # Create output directory
    output_dir = tempfile.mkdtemp(prefix="are_multi_agent_run_")
    
    try:
        # Run scenarios
        result_file = run_scenarios(scenario_ids, output_dir)
        
        if not result_file:
            print("No results file generated. Exiting.")
            sys.exit(1)
        
        # Parse results
        print("\nParsing results...")
        results = parse_results(result_file)
        
        # Print summary
        print_summary(results, scenario_ids)
        
        print(f"Results file: {os.path.abspath(result_file)}")
        print(f"Output directory: {os.path.abspath(output_dir)}")
        print("\nDone.")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

