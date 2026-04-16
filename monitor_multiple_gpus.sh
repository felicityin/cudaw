#!/bin/bash
# File name: monitor_multiple_gpus.sh
# Monitor utilization of multiple specified GPU cards
# Usage: ./monitor_multiple_gpus.sh -g 0,1,2,3

# Default GPU to monitor: 0
GPU_INDICES="0"
OUTPUT_PREFIX="gpu_monitor"
INTERVAL=0.3
DURATION=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            GPU_INDICES="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -g, --gpus INDICES   Specify GPU indices, comma-separated (default: 0)"
            echo "  -o, --output PREFIX  Output file prefix (default: gpu_monitor)"
            echo "  -i, --interval SEC   Monitoring interval in seconds (default: 5)"
            echo "  -d, --duration SEC   Monitoring duration in seconds, 0 for infinite (default: 0)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use $0 -h for help"
            exit 1
            ;;
    esac
done

# Validate GPU index format
IFS=',' read -ra GPU_ARRAY <<< "$GPU_INDICES"
for gpu in "${GPU_ARRAY[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "Error: GPU index must be a number: $gpu"
        exit 1
    fi
    
    # Check if GPU exists
    if ! nvidia-smi -i $gpu --query-gpu=name --format=csv,noheader &>/dev/null; then
        echo "Error: GPU $gpu does not exist or is inaccessible"
        echo "Available GPU list:"
        nvidia-smi --query-gpu=index,name --format=csv
        exit 1
    fi
done

echo "Will monitor the following GPUs: ${GPU_INDICES}"
echo "Press Ctrl+C to stop monitoring"

# Create separate log files for each GPU
declare -A OUTPUT_FILES
for gpu in "${GPU_ARRAY[@]}"; do
    OUTPUT_FILE="${OUTPUT_PREFIX}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).csv"
    OUTPUT_FILES[$gpu]="$OUTPUT_FILE"
    
    # Create CSV file and write header
    echo "timestamp,gpu_index,gpu_name,utilization_gpu(%),utilization_memory(%),memory_used(MB),memory_total(MB),temperature(C),power_draw(W),power_limit(W),process_count" > "$OUTPUT_FILE"
    echo "Log for GPU $gpu will be saved to: $OUTPUT_FILE"
done

# Create summary log file
SUMMARY_FILE="${OUTPUT_PREFIX}_summary_$(date +%Y%m%d_%H%M%S).csv"
echo "timestamp,${GPU_INDICES//,/ }" | sed 's/ /_util, /g; s/$/_util/' > "$SUMMARY_FILE"
echo "Summary data will be saved to: $SUMMARY_FILE"

# Main monitoring loop
start_time=$(date +%s)
count=0

trap 'echo -e "\nMonitoring interrupted by user"; for gpu in "${!OUTPUT_FILES[@]}"; do echo "GPU $gpu: ${OUTPUT_FILES[$gpu]}"; done; exit 0' INT

while true; do
    # Check if duration limit is reached
    if [ $DURATION -gt 0 ]; then
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        [ $elapsed -ge $DURATION ] && break
    fi
    
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    count=$((count + 1))
    
    # Summary data line
    summary_line="$timestamp"
    
    # Monitor each specified GPU
    for gpu in "${GPU_ARRAY[@]}"; do
        # Get GPU information
        gpu_info=$(nvidia-smi -i $gpu \
            --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit \
            --format=csv,noheader,nounits 2>/dev/null)
        
        # Get process count
        process_count=$(nvidia-smi pmon -c 1 -i $gpu 2>/dev/null | grep -v "^#" | grep -v "^\s*$" | wc -l)
        
        if [ -n "$gpu_info" ]; then
            # Write to individual GPU log
            echo "$timestamp,$gpu_info,$process_count" >> "${OUTPUT_FILES[$gpu]}"
            
            # Extract GPU utilization for summary
            gpu_util=$(echo "$gpu_info" | cut -d',' -f3)
            summary_line="$summary_line,$gpu_util"
            
            # Display status every 1st iteration
            if [ $((count % 1)) -eq 0 ]; then
                IFS=',' read -r idx name util mem_used mem_total temp power_draw power_limit <<< "$gpu_info"
                echo "[$timestamp] GPU${idx}: ${util}% usage, ${mem_used}/${mem_total}MB memory"
            fi
        else
            echo "[$timestamp] Error: Failed to get info for GPU $gpu" | tee -a "${OUTPUT_PREFIX}_error.log"
            summary_line="$summary_line,N/A"
        fi
    done
    
    # Write summary data
    echo "$summary_line" >> "$SUMMARY_FILE"
    
    # Wait for specified interval
    sleep $INTERVAL
done

echo "Monitoring completed"
for gpu in "${!OUTPUT_FILES[@]}"; do
    echo "GPU $gpu log: ${OUTPUT_FILES[$gpu]}"
done
echo "Summary log: $SUMMARY_FILE"
