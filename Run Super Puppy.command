#!/bin/bash
# Double-click this to start Super Puppy if it isn't running.
# Also useful after an auto-update if the app didn't restart.
PLIST="$HOME/Library/LaunchAgents/com.local-models.menubar.plist"
if [ -f "$PLIST" ]; then
    launchctl bootout gui/$(id -u)/com.local-models.menubar 2>/dev/null
    sleep 1
    launchctl bootstrap gui/$(id -u) "$PLIST"
    echo "Super Puppy started."
else
    echo "ERROR: LaunchAgent not installed. Run install.sh first."
    exit 1
fi
