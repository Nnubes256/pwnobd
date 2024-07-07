# pwnobd

Offensive cybersecurity toolkit for vulnerability analysis and penetration testing of OBD-II devices.

Part of an as-of-yet unpublished paper.

## Get started

The software is packaged as a Python package; thus, you can use [*pipx*](https://github.com/pypa/pipx) to quickly install the software in a controlled environment:

```bash
pipx install git+https://github.com/Nnubes256/pwnobd.git
```

Once installed, run the software with:
```bash
pwnobd
```

Use `help` to discover the available commands. Command history is saved in `~/.pwnobd_history`.

## Development setup

```bash
# Clone the repository
git clone https://github.com/Nnubes256/pwnobd.git

# Move into the project's root directory
cd pwnobd

# Initialize a virtual environment
# If you're using "pyenv", this will use Python 3.11 automatically
python -m venv .env

# Activate the virtual environment
source .env/bin/activate

# Install the project as a package into the virtual environment
pip install -e .
```

## Adding new functionality

Most functionality is dynamically registered onto pwnobd through the use of decorators.

### Attacks

Located in `src/pwnobd/modules/attacks/`. See the [example](src/pwnobd/modules/attacks/example.py) to understand how to develop an attack and allow the user to parametrize it.

The lifecycle of an attack is as follows:
- `def precheck(**kwargs)`: called every time an option is set for this attack. Allows to implement custom parameter validation beyond what `OPTIONS` provides.
- `def __init__(self, arg1: type1, arg2: type2, ...)`: first step of attack launch; ingest the parameters you specified within `OPTIONS` here.
- `async def setup(self)`: second step of attack launch, perform initialization here. Input is blocked and the attack task is not created until this coroutine finishes, so you may take advantage of interactivity here. Raise `pwnobd.exceptions.PwnObdException` with a custom message if something goes wrong in order to abort the attack's launch.
- `async def run(self, ctx: WorkTaskContext, devices: dict[int, Device])`: actual attack implementation here. Runs within a task. Devices are provided to you pre-locked and ready to use as part of `devices`. **Note:** if you launch tasks which use a device provided here, ensure they finish by the time this coroutine returns; otherwise undefined behavior may happen.

Once your `Attack` subclass is ready, annotate it with `@attack`; as soon as the corresponding Python file is loaded by `pwnobd`, the attack will be automatically registered on startup.

The following parameter types are supported:
- `int`
- `float`
- `str`
    - If a parameter is supposed to be a path, you can add `"hint": "path"` along with `"type": "str"`; this will enable auto-completion of paths within the UI.
- An array of any of the above.

The class constant `DEVICE_REQUIREMENTS` contains the minimum set of classes the device driver must subclass in order for the attack to run. This is validated at runtime by the core logic before the attack class is instanced.

### Device drivers and scanners

Located in `src/pwnobd/modules/devices/`.

A device driver is a class that at least subclasses `pwnobd.core.Device`, plus any common interfaces it may implement (`SendCan`, `Reset`, `Interactive`, etc.).

The lifecycle of a device is as follows.
- The device is instantiated using `__init__`, options as required. Provide a docstring here with the initialization parameters you want to accept; the method's signature and docstring will automatically be parsed upon registration and exposed accordingly to the user.
- `async def connect(self)` is then called; the connection with the device is set up and the device is initialized here.
- Once the connection is ready, `async def handle(self)` is ran as a task. You may handle communications with the device here. Once the task returns, the connection is marked as destroyed and can no longer be used.

An scanner is a class which subclasses `Scanner` or `LeafScanner`.
- A `Scanner` does **not** return instantiable `Devices`s; it may instead return anything. `Scanner`s are not explicitly called when running `recon scan`, and instead are used as a base to implement other scanners. For example, `pwnobd.modules.bluetooth.BluetoothScanner` performs a Bluetooth scan using [*bleak*](https://bleak.readthedocs.io/en/latest/)'s [`BleakScanner`](https://bleak.readthedocs.io/en/latest/api/scanner.html) and returns its results as-is.
- A `LeafScanner` returns a list of objects that subclass `ScannedDevice`. These objects can in turn be used to directly instantiate a device driver of a given type with the required parameters already filled in (which is what happens when `connect --scanned N` is executed). They often rely on upstream `Scanner`s whose results they filter in order to find devices compatible with a given device driver; this is done as follows:

  ```python
  class BluetoothThingymabobScanResult(ScannedDevice):
      # TODO implement
      #   name(self) -> str
      #   device_type(self) -> str
      #   create_device(self) -> Device
      pass

  @scanner("thingymabob")
  class BluetoothThingymabobScanner(LeafScanner):
      async def scan(self, ctx: ScanContext):
          # Retrieve `BluetoothScanner` and ask it to scan for Bluetooth devices.
          devices = await ctx.get_scanner(BluetoothScanner).scan(ctx)

          # ... do something with the returned devices...

          return [
            BluetoothThingymabobScanResult(...),
            BluetoothThingymabobScanResult(...),
            # ...
          ]
  ```

  Results from upstream scanners are cached within a given `recon scan` run; each scanner thus runs only once.

Once your `Device` subclass is ready, annotate it with `@device("device_name")`; as soon as the corresponding Python file is loaded by `pwnobd`, the attack will be automatically registered on startup. Same goes for scanners, use `@scanner("scanner_name")`.

### Commands

TODO