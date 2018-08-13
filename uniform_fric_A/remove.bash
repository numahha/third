#!/bin/bash

find ./ -name '*~' -exec rm {} \;
find ./ -name '*.dvi' -exec rm {} \;
find ./ -name '*.aux' -exec rm {} \;
find ./ -name '*memo.log' -exec rm {} \;
