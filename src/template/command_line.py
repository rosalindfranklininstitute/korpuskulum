#   Copyright 2024 Rosalind Franklin Institute
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from argparse import ArgumentParser

import template


def main(args=None):
    parser = ArgumentParser(description="Add two values")

    parser.add_argument(
        "-a", dest="a", type=float, required=True, help="The first value"
    )

    parser.add_argument(
        "-b", dest="b", type=float, required=True, help="The second value"
    )

    args = parser.parse_args(args)

    print(template.example(args.a, args.b))
