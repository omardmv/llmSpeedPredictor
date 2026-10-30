#!/bin/zsh
# Build Inference Lab and open it as a normal Mac app.
set -e
cd "$(dirname "$0")"
swift build -c release
APP=build/InferenceLab.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp .build/release/InferenceLab "$APP/Contents/MacOS/"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Inference Lab</string>
  <key>CFBundleIdentifier</key><string>local.inferencelab</string>
  <key>CFBundleExecutable</key><string>InferenceLab</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
open "$APP"
