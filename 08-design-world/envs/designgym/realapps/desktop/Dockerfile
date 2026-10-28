# A Linux desktop with real design apps (LibreOffice Impress and Draw: Draw comes with Impress; Inkscape) for the DesignGym GUI surface.
# agent.py serves screenshots and mouse/keyboard input for one X display, and sets each app's episode up and reads
# its document back, so the env's GUI tools drive these apps exactly as they drive the browser editors.
# A plugin with more apps builds its own image (with those apps) and registers it (realapps/registry.py); its
# agent-side modules go in /opt/agent/apps.
FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 DISPLAY=:1
RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb xdotool x11-utils openbox dbus-x11 \
      at-spi2-core gir1.2-atspi-2.0 \
      libreoffice-impress libreoffice-gtk3 python3-uno \
      inkscape \
      python3 python3-pil python3-numpy fonts-dejavu-core fontconfig ca-certificates procps \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 user && mkdir -p /work && chown user /work
COPY agent.py /opt/agent/agent.py
USER user
WORKDIR /home/user
# the task pack's fonts are mounted at /fonts and registered at start, so the apps offer the same families
EXPOSE 5005
CMD ["python3", "/opt/agent/agent.py", "--port", "5005"]
