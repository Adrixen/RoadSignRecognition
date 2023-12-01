Installing OpenCV, necessary dependencies and QOL features on Raspberry Pi 4B is as follows:
1. Enable I2C and VNC support in Preferences>Raspberry Pi Configuration>Interfaces.
2. Make sure to log out and log in at least once from RaspberryPi as it will
create necessary bootstrap files containing environment PATHS. (e.g. needed for pip)
3. Update and upgrade packages with:
sudo apt-get update && sudo apt-get upgrade
4. Install dependencies with:
sudo apt install -y build-essential cmake pkg-config libjpeg-dev libtiff5-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 libqt5gui5 libqt5webkit5 libqt5test5 python3-pyqt5 python3-dev
5. Install PiCamera module with:
pip install "picamera[array]"
6. Upgrade pip with:
pip install --upgrade pip
7. Install openCV with:
pip install opencv-python


Starting the project requires loading libatomic library. 
It needs to be seen by the Linux dynamic linker before Python is loaded.
Command below launches Thonny IDE with libatomic library.

LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1 thonny &

Command below launches Python in terminal with libatomic library.

LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1 python3
