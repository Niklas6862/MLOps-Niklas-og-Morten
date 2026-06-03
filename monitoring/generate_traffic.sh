#!/usr/bin/env bash
# Sends 2000 requests to populate Grafana metrics.
# Distribution mirrors real field conditions: 40% angular_leaf_spot, 35% bean_rust, 25% healthy.
BASE_URL="${1:-http://localhost:8000}"

echo "Sending 2000 requests to $BASE_URL/predict ..."

for i in $(seq 1 800); do
    curl -s -X POST "$BASE_URL/predict" \
         -H "Content-Type: application/json" \
         -d "{\"image_id\": \"angular_leaf_spot_$(printf '%04d' $i)\"}" > /dev/null
done

for i in $(seq 1 700); do
    curl -s -X POST "$BASE_URL/predict" \
         -H "Content-Type: application/json" \
         -d "{\"image_id\": \"bean_rust_$(printf '%04d' $i)\"}" > /dev/null
done

for i in $(seq 1 500); do
    curl -s -X POST "$BASE_URL/predict" \
         -H "Content-Type: application/json" \
         -d "{\"image_id\": \"healthy_$(printf '%04d' $i)\"}" > /dev/null
done

echo "Done. Open Grafana at http://localhost:3000"
