# Copyright (c) 2016, The developers of the Stanford CRN
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of crn_base nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This Dockerfile is to be built for archiving in Docker hub and use through
# the agave platform.

FROM oesteban/crn_nipype

ARG GIT_BRANCH
ENV GIT_BRANCH ${GIT_BRANCH:-docker}

ARG GIT_URL
ENV GIT_URL ${GIT_URL:-"https://github.com/poldracklab/preprocessing-workflow.git"}

WORKDIR /root/src
RUN git clone $GIT_URL && \
    cd preprocessing-workflow && \
    git checkout $GIT_BRANCH && \
    source activate crnenv && \
    python setup.py install
    
ADD files/run_* /usr/bin/
RUN chmod +x /usr/bin/run_*

RUN groupadd -g 1002 agave && \
    groupadd -g 1001 docker && \
    useradd -m -G agave,docker -s /bin/bash -u 1001 ubuntu
USER ubuntu

ENTRYPOINT ["/usr/bin/run_fmriprep"]
CMD ["--help"]
