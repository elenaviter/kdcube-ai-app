# Create a file: ~/chrome-debug.sh
#!/bin/bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --user-data-dir=/tmp/chrome-debug-profile --remote-debugging-port=62006 "$@"