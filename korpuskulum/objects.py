#   Copyright 2025 Rosalind Franklin Institute
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


from dataclasses import dataclass
import typing


@dataclass()
class Config:
    pixel_size_nm: typing.Optional[float] = None
    dist_range: typing.Optional[list] = None
    coords_files: typing.Optional[list] = None
    membrane_files: typing.Optional[list] = None
    order: typing.Optional[str] = None
