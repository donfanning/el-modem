# Linux virtual serial prerequisite

The modem executable runs unprivileged. Full virtual-serial support requires
serialsim at revision `60c0f40d884d92b80442857ed1b659f60294d9fb` with the included
GPL-2.0+ patch. The modem does not install or load a module automatically.

The patch gates the transfer thread with `uart_tx_stopped`, makes an open port
exclusive, and adds read-only remote open-count ioctl `0x800454ec`. Normal users
cannot replace an application which owns the port; privileged processes retain
Linux's standard exclusive-TTY bypass. The driver controls signals separately
rather than tying carrier to DTR through null-modem wiring.

Build for the running kernel (requires git, make, compiler and kernel headers):

```sh
packaging/serial/build-driver.sh /absolute/new/serialsim-build
```

An administrator then installs/signs the module using serialsim's installation
instructions. Secure Boot may require enrollment of a signing key. Load with
an appropriate dynamic-port capacity, for example:

```sh
sudo insmod /absolute/new/serialsim-build/serialsim.ko nr_pipe_ports=0 nr_echo_ports=0 nr_dyn_pipe_ports=64
```

Create a serial-access group and add the intended modem/application users. Use
udev rules assigning `/dev/ttyPipe*` to that group with mode 0660, reload rules,
and apply them to existing devices. Do not use world-writable device permissions.
The supervisor reports unavailable devices or exhausted driver capacity.

Qualification (run as an ordinary permitted user):

```sh
EL_MODEM_TEST_DRIVER=1 PYTHONPATH=lab/asterisk python -m unittest test_modem_serial_driver -v
```

Tested initially on Linux 6.8.0-142-generic. Other kernels need the same tests;
compilation alone does not establish signal/flow-control behavior. Both byte
paths, CTS/RTS pause/resume, DTR, DCD/DSR/RI, exclusivity and device cleanup are
release requirements. PTY/TCP are explicit alternatives with different signal
capabilities, not transparent replacements when this prerequisite is missing.
