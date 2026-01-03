#!/bin/bash

# Have to start ovs service first
service openvswitch-switch start

uv run route_agent.py "${@}"