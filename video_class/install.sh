#!/bin/bash

echo "start..."
cd ~/.hmpclient/bin/ &&
wget -q http://10.197.94.11:8000/wflow_cli/latest_version/wflow_cli.tar.gz -O wflow_cli.tar.gz &&
echo "download!"
tar zxf wflow_cli.tar.gz &&
rm wflow_cli.tar.gz &&
echo "SUCCESS!"
