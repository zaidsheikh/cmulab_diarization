## Speaker Diarization plugin for the [CMU Linguistic Annotation Backend](https://github.com/neulab/cmulab/)

### Install

```
python3 -m pip install git+https://github.com/zaidsheikh/cmulab_diarization
```

This package registers itself as a plugin for [CMULAB](https://github.com/neulab/cmulab/) (CMU Linguistic Annotation Backend) by [registering an entrypoint](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#dynamic-discovery-of-services-and-plugins) via setuptools.

```
[options.entry_points]
cmulab.plugins =
    diarization = cmulab_diarization:get_results
```
