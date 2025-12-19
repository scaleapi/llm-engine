#!/bin/bash

# Quick Rollback Script
# This will rollback the Helm release to revision 1

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${RED}🚨 ROLLBACK INITIATED${NC}"
echo -e "${YELLOW}This will rollback launch-inference to revision 1${NC}"
echo

read -p "Are you sure you want to rollback? (yes/no) " -r
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo -e "${RED}Rollback cancelled${NC}"
    exit 1
fi

echo -e "${BLUE}1. Rolling back Helm release...${NC}"
helm rollback launch-inference 1 -n launch

echo -e "${BLUE}2. Monitoring model-engine deployment...${NC}"
kubectl rollout status deployment/model-engine -n launch --timeout=300s

echo -e "${BLUE}3. Monitoring endpoint-builder deployment...${NC}"
kubectl rollout status deployment/model-engine-endpoint-builder -n launch --timeout=300s

echo -e "${GREEN}✅ Rollback completed!${NC}"
echo

echo -e "${BLUE}Verifying pods...${NC}"
kubectl get pods -n launch | grep model-engine

echo
echo -e "${BLUE}Verifying image...${NC}"
CURRENT_IMAGE=$(kubectl get deployment model-engine -n launch -o jsonpath='{.spec.template.spec.containers[0].image}')
EXPECTED_IMAGE="022465994601.dkr.ecr.us-west-2.amazonaws.com/egp-mirror-int/model-engine:6e35c71cf82622fe2ad6e745728a65a1ff6f3984"

if [ "$CURRENT_IMAGE" = "$EXPECTED_IMAGE" ]; then
    echo -e "${GREEN}✅ Image verified: $CURRENT_IMAGE${NC}"
else
    echo -e "${RED}❌ Image mismatch!${NC}"
    echo -e "   Expected: $EXPECTED_IMAGE"
    echo -e "   Got: $CURRENT_IMAGE"
fi

echo
echo -e "${GREEN}🎉 Rollback successful!${NC}"
