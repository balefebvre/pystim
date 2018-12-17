import importlib
import json
import os
import shutil


environment_variable_name = 'PYSTIMPATH'


def list_stimuli():

    stimuli = [
        'dg',
        'fipwc',
    ]
    # TODO list the stimuli.

    return stimuli


def get_default_configuration_path():

    path = os.path.join("~", ".config", "pystim")
    path = os.path.expanduser(path)

    path = os.getenv(environment_variable_name, path)

    return path


def configure(args):

    _ = args
    path = get_default_configuration_path()
    message = "Default configuration path: {}".format(path)
    print(message)

    return


def initialize(args):

    _ = args

    # Create the default configuration path (if necessary).
    path = get_default_configuration_path()
    if not os.path.isdir(path):
        os.makedirs(path)

    # List the stimuli.
    stimuli = list_stimuli()
    # For each stimulus...
    for stimulus in stimuli:
        # Get the corresponding configuration.
        module_name = 'pystim.{}'.format(stimulus)
        module = importlib.import_module(module_name)
        configuration = module.default_configuration
        # Store the configuration in a file.
        output_filename = "{}.json".format(stimulus)
        output_path = os.path.join(path, output_filename)
        with open(output_path, mode='w') as output_file:
            json.dump(configuration, output_file, indent=4)

    return


def reinitialize(args):

    path = get_default_configuration_path()
    shutil.rmtree(path)
    initialize(args)

    return


def load_configuration(path):

    try:
        with open(path, mode='r') as file:
            configuration = json.load(file)
    except FileNotFoundError:
        configuration = {}

    return configuration


def load_global_configuration(name):

    path = get_default_configuration_path()
    filename = "{}.json".format(name)
    path = os.path.join(path, filename)
    configuration = load_configuration(path)

    return configuration


def load_local_configuration(name):

    path = os.getcwd()
    filename = "{}.json".format(name)
    path = os.path.join(path, filename)
    configuration = load_configuration(path)

    return configuration


def handle_arguments_and_configurations(name, args):

    global_configuration = load_global_configuration(name)
    local_configuration = load_local_configuration(name)
    arguments = vars(args)

    assert 'func' not in global_configuration
    assert 'func' not in local_configuration

    configuration = {}
    configuration.update(global_configuration)
    configuration.update(local_configuration)
    configuration.update(arguments)  # TODO handle nested fields.

    return configuration
