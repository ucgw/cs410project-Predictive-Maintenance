import sys
from configobj import (
    ConfigObj,
    ConfigObjError,
    flatten_errors
)
from validate import (
    Validator,
    ValidateError,
    VdtTypeError
)

def error_exit(msg):
    sys.stderr.write("ModelConfig Error: {msg}\n".format(msg=str(msg)))
    sys.exit(1)

def validate_config(config):
    validator = Validator()
    try:
        results = config.validate(validator, preserve_errors=True)
    except ValidateError as valerror:
        error_exit(valerror)

    if isinstance(results, dict):
        conferr = "config validation errors caught!"
        for (section_list, key, _) in flatten_errors(config, results):
            if section_list is not None:
                sections = ','.join(section_list)
                if key is not None:
                    conferr = "{conferr}\n  section(s)={sections} key={key} failed validation".format(conferr=conferr, sections=sections, key=key)
                else:
                    conferr = "{conferr}\n  section(s): {sections} missing".format(conferr=conferr, sections=sections)
        error_exit(conferr)

def build_config(config_file, spec_file):
    try:
        config = ConfigObj(config_file, configspec=spec_file, raise_errors=True, file_error=True, interpolation=False)
    except (ConfigObjError, IOError, Exception) as conferr:
        error_exit(conferr)
    finally:
        validate_config(config)
        return config
