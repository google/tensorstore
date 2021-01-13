"""Defines third-party bazel repos for Python packages fetched with pip."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_absl_py()
    repo_pypa_alabaster()
    repo_pypa_apache_beam()
    repo_pypa_appnope()
    repo_pypa_astor()
    repo_pypa_atomicwrites()
    repo_pypa_attrs()
    repo_pypa_avro_python3()
    repo_pypa_babel()
    repo_pypa_backcall()
    repo_pypa_certifi()
    repo_pypa_chardet()
    repo_pypa_colorama()
    repo_pypa_crcmod()
    repo_pypa_decorator()
    repo_pypa_dill()
    repo_pypa_docutils()
    repo_pypa_fastavro()
    repo_pypa_future()
    repo_pypa_gin_config()
    repo_pypa_grpcio()
    repo_pypa_hdfs()
    repo_pypa_httplib2()
    repo_pypa_idna()
    repo_pypa_imagesize()
    repo_pypa_importlib_metadata()
    repo_pypa_iniconfig()
    repo_pypa_ipython()
    repo_pypa_ipython_genutils()
    repo_pypa_jedi()
    repo_pypa_jinja2()
    repo_pypa_jsonpointer()
    repo_pypa_jsonschema()
    repo_pypa_markupsafe()
    repo_pypa_mock()
    repo_pypa_numpy()
    repo_pypa_oauth2client()
    repo_pypa_packaging()
    repo_pypa_parso()
    repo_pypa_pexpect()
    repo_pypa_pickleshare()
    repo_pypa_pluggy()
    repo_pypa_prompt_toolkit()
    repo_pypa_protobuf()
    repo_pypa_ptyprocess()
    repo_pypa_py()
    repo_pypa_pyarrow()
    repo_pypa_pyasn1()
    repo_pypa_pyasn1_modules()
    repo_pypa_pydot()
    repo_pypa_pygments()
    repo_pypa_pymongo()
    repo_pypa_pyparsing()
    repo_pypa_pyrsistent()
    repo_pypa_pytest()
    repo_pypa_pytest_asyncio()
    repo_pypa_python_dateutil()
    repo_pypa_pytz()
    repo_pypa_pyyaml()
    repo_pypa_requests()
    repo_pypa_rsa()
    repo_pypa_setuptools()
    repo_pypa_six()
    repo_pypa_snowballstemmer()
    repo_pypa_sphinx()
    repo_pypa_sphinx_rtd_theme()
    repo_pypa_sphinxcontrib_applehelp()
    repo_pypa_sphinxcontrib_devhelp()
    repo_pypa_sphinxcontrib_htmlhelp()
    repo_pypa_sphinxcontrib_jsmath()
    repo_pypa_sphinxcontrib_qthelp()
    repo_pypa_sphinxcontrib_serializinghtml()
    repo_pypa_toml()
    repo_pypa_traitlets()
    repo_pypa_typing_extensions()
    repo_pypa_urllib3()
    repo_pypa_wcwidth()
    repo_pypa_wheel()
    repo_pypa_zipp()

def repo_pypa_absl_py():
    repo_pypa_six()
    maybe(
        third_party_python_package,
        name = "pypa_absl_py",
        target = "absl_py",
        requirement = "absl-py==0.11.0",
        deps = [
            "@pypa_six//:six",
        ],
    )

def repo_pypa_alabaster():
    maybe(
        third_party_python_package,
        name = "pypa_alabaster",
        target = "alabaster",
        requirement = "alabaster==0.7.12",
    )

def repo_pypa_apache_beam():
    repo_pypa_avro_python3()
    repo_pypa_crcmod()
    repo_pypa_dill()
    repo_pypa_fastavro()
    repo_pypa_future()
    repo_pypa_grpcio()
    repo_pypa_hdfs()
    repo_pypa_httplib2()
    repo_pypa_mock()
    repo_pypa_numpy()
    repo_pypa_oauth2client()
    repo_pypa_protobuf()
    repo_pypa_pyarrow()
    repo_pypa_pydot()
    repo_pypa_pymongo()
    repo_pypa_python_dateutil()
    repo_pypa_pytz()
    repo_pypa_requests()
    repo_pypa_typing_extensions()
    maybe(
        third_party_python_package,
        name = "pypa_apache_beam",
        target = "apache_beam",
        requirement = "apache-beam==2.27.0",
        deps = [
            "@pypa_avro_python3//:avro_python3",
            "@pypa_crcmod//:crcmod",
            "@pypa_dill//:dill",
            "@pypa_fastavro//:fastavro",
            "@pypa_future//:future",
            "@pypa_grpcio//:grpcio",
            "@pypa_hdfs//:hdfs",
            "@pypa_httplib2//:httplib2",
            "@pypa_mock//:mock",
            "@pypa_numpy//:numpy",
            "@pypa_oauth2client//:oauth2client",
            "@pypa_protobuf//:protobuf",
            "@pypa_pyarrow//:pyarrow",
            "@pypa_pydot//:pydot",
            "@pypa_pymongo//:pymongo",
            "@pypa_python_dateutil//:python_dateutil",
            "@pypa_pytz//:pytz",
            "@pypa_requests//:requests",
            "@pypa_typing_extensions//:typing_extensions",
        ],
    )

def repo_pypa_appnope():
    maybe(
        third_party_python_package,
        name = "pypa_appnope",
        target = "appnope",
        requirement = "appnope==0.1.2",
    )

def repo_pypa_astor():
    maybe(
        third_party_python_package,
        name = "pypa_astor",
        target = "astor",
        requirement = "astor==0.8.1",
    )

def repo_pypa_atomicwrites():
    maybe(
        third_party_python_package,
        name = "pypa_atomicwrites",
        target = "atomicwrites",
        requirement = "atomicwrites==1.4.0",
    )

def repo_pypa_attrs():
    maybe(
        third_party_python_package,
        name = "pypa_attrs",
        target = "attrs",
        requirement = "attrs==20.3.0",
    )

def repo_pypa_avro_python3():
    maybe(
        third_party_python_package,
        name = "pypa_avro_python3",
        target = "avro_python3",
        requirement = "avro-python3==1.10.1",
    )

def repo_pypa_babel():
    repo_pypa_pytz()
    maybe(
        third_party_python_package,
        name = "pypa_babel",
        target = "babel",
        requirement = "babel==2.9.0",
        deps = [
            "@pypa_pytz//:pytz",
        ],
    )

def repo_pypa_backcall():
    maybe(
        third_party_python_package,
        name = "pypa_backcall",
        target = "backcall",
        requirement = "backcall==0.2.0",
    )

def repo_pypa_certifi():
    maybe(
        third_party_python_package,
        name = "pypa_certifi",
        target = "certifi",
        requirement = "certifi==2020.12.5",
    )

def repo_pypa_chardet():
    maybe(
        third_party_python_package,
        name = "pypa_chardet",
        target = "chardet",
        requirement = "chardet==4.0.0",
    )

def repo_pypa_colorama():
    maybe(
        third_party_python_package,
        name = "pypa_colorama",
        target = "colorama",
        requirement = "colorama==0.4.4",
    )

def repo_pypa_crcmod():
    maybe(
        third_party_python_package,
        name = "pypa_crcmod",
        target = "crcmod",
        requirement = "crcmod==1.7",
    )

def repo_pypa_decorator():
    maybe(
        third_party_python_package,
        name = "pypa_decorator",
        target = "decorator",
        requirement = "decorator==4.4.2",
    )

def repo_pypa_dill():
    maybe(
        third_party_python_package,
        name = "pypa_dill",
        target = "dill",
        requirement = "dill==0.3.3",
    )

def repo_pypa_docutils():
    maybe(
        third_party_python_package,
        name = "pypa_docutils",
        target = "docutils",
        requirement = "docutils==0.16",
    )

def repo_pypa_fastavro():
    maybe(
        third_party_python_package,
        name = "pypa_fastavro",
        target = "fastavro",
        requirement = "fastavro==1.2.3",
    )

def repo_pypa_future():
    maybe(
        third_party_python_package,
        name = "pypa_future",
        target = "future",
        requirement = "future==0.18.2",
    )

def repo_pypa_gin_config():
    maybe(
        third_party_python_package,
        name = "pypa_gin_config",
        target = "gin_config",
        requirement = "gin-config==0.4.0",
    )

def repo_pypa_grpcio():
    repo_pypa_six()
    maybe(
        third_party_python_package,
        name = "pypa_grpcio",
        target = "grpcio",
        requirement = "grpcio==1.34.0",
        deps = [
            "@pypa_six//:six",
        ],
    )

def repo_pypa_hdfs():
    maybe(
        third_party_python_package,
        name = "pypa_hdfs",
        target = "hdfs",
        requirement = "hdfs==2.5.8",
    )

def repo_pypa_httplib2():
    maybe(
        third_party_python_package,
        name = "pypa_httplib2",
        target = "httplib2",
        requirement = "httplib2==0.18.1",
    )

def repo_pypa_idna():
    maybe(
        third_party_python_package,
        name = "pypa_idna",
        target = "idna",
        requirement = "idna==3.1",
    )

def repo_pypa_imagesize():
    maybe(
        third_party_python_package,
        name = "pypa_imagesize",
        target = "imagesize",
        requirement = "imagesize==1.2.0",
    )

def repo_pypa_importlib_metadata():
    repo_pypa_typing_extensions()
    repo_pypa_zipp()
    maybe(
        third_party_python_package,
        name = "pypa_importlib_metadata",
        target = "importlib_metadata",
        requirement = "importlib-metadata==3.3.0",
        deps = [
            "@pypa_typing_extensions//:typing_extensions",
            "@pypa_zipp//:zipp",
        ],
    )

def repo_pypa_iniconfig():
    maybe(
        third_party_python_package,
        name = "pypa_iniconfig",
        target = "iniconfig",
        requirement = "iniconfig==1.1.1",
    )

def repo_pypa_ipython():
    repo_pypa_appnope()
    repo_pypa_backcall()
    repo_pypa_colorama()
    repo_pypa_decorator()
    repo_pypa_jedi()
    repo_pypa_pexpect()
    repo_pypa_pickleshare()
    repo_pypa_prompt_toolkit()
    repo_pypa_pygments()
    repo_pypa_setuptools()
    repo_pypa_traitlets()
    maybe(
        third_party_python_package,
        name = "pypa_ipython",
        target = "ipython",
        requirement = "ipython==7.19.0",
        deps = [
            "@pypa_appnope//:appnope",
            "@pypa_backcall//:backcall",
            "@pypa_colorama//:colorama",
            "@pypa_decorator//:decorator",
            "@pypa_jedi//:jedi",
            "@pypa_pexpect//:pexpect",
            "@pypa_pickleshare//:pickleshare",
            "@pypa_prompt_toolkit//:prompt_toolkit",
            "@pypa_pygments//:pygments",
            "@pypa_setuptools//:setuptools",
            "@pypa_traitlets//:traitlets",
        ],
    )

def repo_pypa_ipython_genutils():
    maybe(
        third_party_python_package,
        name = "pypa_ipython_genutils",
        target = "ipython_genutils",
        requirement = "ipython-genutils==0.2.0",
    )

def repo_pypa_jedi():
    repo_pypa_parso()
    maybe(
        third_party_python_package,
        name = "pypa_jedi",
        target = "jedi",
        requirement = "jedi==0.18.0",
        deps = [
            "@pypa_parso//:parso",
        ],
    )

def repo_pypa_jinja2():
    repo_pypa_markupsafe()
    maybe(
        third_party_python_package,
        name = "pypa_jinja2",
        target = "jinja2",
        requirement = "jinja2==2.11.2",
        deps = [
            "@pypa_markupsafe//:markupsafe",
        ],
    )

def repo_pypa_jsonpointer():
    maybe(
        third_party_python_package,
        name = "pypa_jsonpointer",
        target = "jsonpointer",
        requirement = "jsonpointer==2.0",
    )

def repo_pypa_jsonschema():
    repo_pypa_attrs()
    repo_pypa_importlib_metadata()
    repo_pypa_pyrsistent()
    repo_pypa_setuptools()
    repo_pypa_six()
    maybe(
        third_party_python_package,
        name = "pypa_jsonschema",
        target = "jsonschema",
        requirement = "jsonschema==3.2.0",
        deps = [
            "@pypa_attrs//:attrs",
            "@pypa_importlib_metadata//:importlib_metadata",
            "@pypa_pyrsistent//:pyrsistent",
            "@pypa_setuptools//:setuptools",
            "@pypa_six//:six",
        ],
    )

def repo_pypa_markupsafe():
    maybe(
        third_party_python_package,
        name = "pypa_markupsafe",
        target = "markupsafe",
        requirement = "markupsafe==1.1.1",
    )

def repo_pypa_mock():
    maybe(
        third_party_python_package,
        name = "pypa_mock",
        target = "mock",
        requirement = "mock==4.0.3",
    )

def repo_pypa_numpy():
    maybe(
        third_party_python_package,
        name = "pypa_numpy",
        target = "numpy",
        requirement = "numpy==1.19.5",
    )

def repo_pypa_oauth2client():
    repo_pypa_httplib2()
    repo_pypa_pyasn1()
    repo_pypa_pyasn1_modules()
    repo_pypa_rsa()
    repo_pypa_six()
    maybe(
        third_party_python_package,
        name = "pypa_oauth2client",
        target = "oauth2client",
        requirement = "oauth2client==4.1.3",
        deps = [
            "@pypa_httplib2//:httplib2",
            "@pypa_pyasn1//:pyasn1",
            "@pypa_pyasn1_modules//:pyasn1_modules",
            "@pypa_rsa//:rsa",
            "@pypa_six//:six",
        ],
    )

def repo_pypa_packaging():
    repo_pypa_pyparsing()
    maybe(
        third_party_python_package,
        name = "pypa_packaging",
        target = "packaging",
        requirement = "packaging==20.8",
        deps = [
            "@pypa_pyparsing//:pyparsing",
        ],
    )

def repo_pypa_parso():
    maybe(
        third_party_python_package,
        name = "pypa_parso",
        target = "parso",
        requirement = "parso==0.8.1",
    )

def repo_pypa_pexpect():
    repo_pypa_ptyprocess()
    maybe(
        third_party_python_package,
        name = "pypa_pexpect",
        target = "pexpect",
        requirement = "pexpect==4.8.0",
        deps = [
            "@pypa_ptyprocess//:ptyprocess",
        ],
    )

def repo_pypa_pickleshare():
    maybe(
        third_party_python_package,
        name = "pypa_pickleshare",
        target = "pickleshare",
        requirement = "pickleshare==0.7.5",
    )

def repo_pypa_pluggy():
    repo_pypa_importlib_metadata()
    maybe(
        third_party_python_package,
        name = "pypa_pluggy",
        target = "pluggy",
        requirement = "pluggy==0.13.1",
        deps = [
            "@pypa_importlib_metadata//:importlib_metadata",
        ],
    )

def repo_pypa_prompt_toolkit():
    repo_pypa_wcwidth()
    maybe(
        third_party_python_package,
        name = "pypa_prompt_toolkit",
        target = "prompt_toolkit",
        requirement = "prompt-toolkit==3.0.10",
        deps = [
            "@pypa_wcwidth//:wcwidth",
        ],
    )

def repo_pypa_protobuf():
    repo_pypa_six()
    maybe(
        third_party_python_package,
        name = "pypa_protobuf",
        target = "protobuf",
        requirement = "protobuf==3.14.0",
        deps = [
            "@pypa_six//:six",
        ],
    )

def repo_pypa_ptyprocess():
    maybe(
        third_party_python_package,
        name = "pypa_ptyprocess",
        target = "ptyprocess",
        requirement = "ptyprocess==0.7.0",
    )

def repo_pypa_py():
    maybe(
        third_party_python_package,
        name = "pypa_py",
        target = "py",
        requirement = "py==1.10.0",
    )

def repo_pypa_pyarrow():
    repo_pypa_numpy()
    maybe(
        third_party_python_package,
        name = "pypa_pyarrow",
        target = "pyarrow",
        requirement = "pyarrow==2.0.0",
        deps = [
            "@pypa_numpy//:numpy",
        ],
    )

def repo_pypa_pyasn1():
    maybe(
        third_party_python_package,
        name = "pypa_pyasn1",
        target = "pyasn1",
        requirement = "pyasn1==0.4.8",
    )

def repo_pypa_pyasn1_modules():
    repo_pypa_pyasn1()
    maybe(
        third_party_python_package,
        name = "pypa_pyasn1_modules",
        target = "pyasn1_modules",
        requirement = "pyasn1-modules==0.2.8",
        deps = [
            "@pypa_pyasn1//:pyasn1",
        ],
    )

def repo_pypa_pydot():
    repo_pypa_pyparsing()
    maybe(
        third_party_python_package,
        name = "pypa_pydot",
        target = "pydot",
        requirement = "pydot==1.4.1",
        deps = [
            "@pypa_pyparsing//:pyparsing",
        ],
    )

def repo_pypa_pygments():
    maybe(
        third_party_python_package,
        name = "pypa_pygments",
        target = "pygments",
        requirement = "pygments==2.7.3",
    )

def repo_pypa_pymongo():
    maybe(
        third_party_python_package,
        name = "pypa_pymongo",
        target = "pymongo",
        requirement = "pymongo==3.11.2",
    )

def repo_pypa_pyparsing():
    maybe(
        third_party_python_package,
        name = "pypa_pyparsing",
        target = "pyparsing",
        requirement = "pyparsing==2.4.7",
    )

def repo_pypa_pyrsistent():
    maybe(
        third_party_python_package,
        name = "pypa_pyrsistent",
        target = "pyrsistent",
        requirement = "pyrsistent==0.17.3",
    )

def repo_pypa_pytest():
    repo_pypa_atomicwrites()
    repo_pypa_attrs()
    repo_pypa_colorama()
    repo_pypa_importlib_metadata()
    repo_pypa_iniconfig()
    repo_pypa_packaging()
    repo_pypa_pluggy()
    repo_pypa_py()
    repo_pypa_toml()
    maybe(
        third_party_python_package,
        name = "pypa_pytest",
        target = "pytest",
        requirement = "pytest==6.2.1",
        deps = [
            "@pypa_atomicwrites//:atomicwrites",
            "@pypa_attrs//:attrs",
            "@pypa_colorama//:colorama",
            "@pypa_importlib_metadata//:importlib_metadata",
            "@pypa_iniconfig//:iniconfig",
            "@pypa_packaging//:packaging",
            "@pypa_pluggy//:pluggy",
            "@pypa_py//:py",
            "@pypa_toml//:toml",
        ],
    )

def repo_pypa_pytest_asyncio():
    repo_pypa_pytest()
    maybe(
        third_party_python_package,
        name = "pypa_pytest_asyncio",
        target = "pytest_asyncio",
        requirement = "pytest-asyncio==0.14.0",
        deps = [
            "@pypa_pytest//:pytest",
        ],
    )

def repo_pypa_python_dateutil():
    repo_pypa_six()
    maybe(
        third_party_python_package,
        name = "pypa_python_dateutil",
        target = "python_dateutil",
        requirement = "python-dateutil==2.8.1",
        deps = [
            "@pypa_six//:six",
        ],
    )

def repo_pypa_pytz():
    maybe(
        third_party_python_package,
        name = "pypa_pytz",
        target = "pytz",
        requirement = "pytz==2020.5",
    )

def repo_pypa_pyyaml():
    maybe(
        third_party_python_package,
        name = "pypa_pyyaml",
        target = "pyyaml",
        requirement = "pyyaml==5.3.1",
    )

def repo_pypa_requests():
    repo_pypa_certifi()
    repo_pypa_chardet()
    repo_pypa_idna()
    repo_pypa_urllib3()
    maybe(
        third_party_python_package,
        name = "pypa_requests",
        target = "requests",
        requirement = "requests==2.25.1",
        deps = [
            "@pypa_certifi//:certifi",
            "@pypa_chardet//:chardet",
            "@pypa_idna//:idna",
            "@pypa_urllib3//:urllib3",
        ],
    )

def repo_pypa_rsa():
    repo_pypa_pyasn1()
    maybe(
        third_party_python_package,
        name = "pypa_rsa",
        target = "rsa",
        requirement = "rsa==4.6",
        deps = [
            "@pypa_pyasn1//:pyasn1",
        ],
    )

def repo_pypa_setuptools():
    maybe(
        third_party_python_package,
        name = "pypa_setuptools",
        target = "setuptools",
        requirement = "setuptools==51.1.1",
    )

def repo_pypa_six():
    maybe(
        third_party_python_package,
        name = "pypa_six",
        target = "six",
        requirement = "six==1.15.0",
    )

def repo_pypa_snowballstemmer():
    maybe(
        third_party_python_package,
        name = "pypa_snowballstemmer",
        target = "snowballstemmer",
        requirement = "snowballstemmer==2.0.0",
    )

def repo_pypa_sphinx():
    repo_pypa_alabaster()
    repo_pypa_babel()
    repo_pypa_colorama()
    repo_pypa_docutils()
    repo_pypa_imagesize()
    repo_pypa_jinja2()
    repo_pypa_packaging()
    repo_pypa_pygments()
    repo_pypa_requests()
    repo_pypa_setuptools()
    repo_pypa_snowballstemmer()
    repo_pypa_sphinxcontrib_applehelp()
    repo_pypa_sphinxcontrib_devhelp()
    repo_pypa_sphinxcontrib_htmlhelp()
    repo_pypa_sphinxcontrib_jsmath()
    repo_pypa_sphinxcontrib_qthelp()
    repo_pypa_sphinxcontrib_serializinghtml()
    maybe(
        third_party_python_package,
        name = "pypa_sphinx",
        target = "sphinx",
        requirement = "sphinx==3.4.3",
        deps = [
            "@pypa_alabaster//:alabaster",
            "@pypa_babel//:babel",
            "@pypa_colorama//:colorama",
            "@pypa_docutils//:docutils",
            "@pypa_imagesize//:imagesize",
            "@pypa_jinja2//:jinja2",
            "@pypa_packaging//:packaging",
            "@pypa_pygments//:pygments",
            "@pypa_requests//:requests",
            "@pypa_setuptools//:setuptools",
            "@pypa_snowballstemmer//:snowballstemmer",
            "@pypa_sphinxcontrib_applehelp//:sphinxcontrib_applehelp",
            "@pypa_sphinxcontrib_devhelp//:sphinxcontrib_devhelp",
            "@pypa_sphinxcontrib_htmlhelp//:sphinxcontrib_htmlhelp",
            "@pypa_sphinxcontrib_jsmath//:sphinxcontrib_jsmath",
            "@pypa_sphinxcontrib_qthelp//:sphinxcontrib_qthelp",
            "@pypa_sphinxcontrib_serializinghtml//:sphinxcontrib_serializinghtml",
        ],
    )

def repo_pypa_sphinx_rtd_theme():
    repo_pypa_sphinx()
    maybe(
        third_party_python_package,
        name = "pypa_sphinx_rtd_theme",
        target = "sphinx_rtd_theme",
        requirement = "sphinx-rtd-theme==0.5.1",
        deps = [
            "@pypa_sphinx//:sphinx",
        ],
    )

def repo_pypa_sphinxcontrib_applehelp():
    maybe(
        third_party_python_package,
        name = "pypa_sphinxcontrib_applehelp",
        target = "sphinxcontrib_applehelp",
        requirement = "sphinxcontrib-applehelp==1.0.2",
    )

def repo_pypa_sphinxcontrib_devhelp():
    maybe(
        third_party_python_package,
        name = "pypa_sphinxcontrib_devhelp",
        target = "sphinxcontrib_devhelp",
        requirement = "sphinxcontrib-devhelp==1.0.2",
    )

def repo_pypa_sphinxcontrib_htmlhelp():
    maybe(
        third_party_python_package,
        name = "pypa_sphinxcontrib_htmlhelp",
        target = "sphinxcontrib_htmlhelp",
        requirement = "sphinxcontrib-htmlhelp==1.0.3",
    )

def repo_pypa_sphinxcontrib_jsmath():
    maybe(
        third_party_python_package,
        name = "pypa_sphinxcontrib_jsmath",
        target = "sphinxcontrib_jsmath",
        requirement = "sphinxcontrib-jsmath==1.0.1",
    )

def repo_pypa_sphinxcontrib_qthelp():
    maybe(
        third_party_python_package,
        name = "pypa_sphinxcontrib_qthelp",
        target = "sphinxcontrib_qthelp",
        requirement = "sphinxcontrib-qthelp==1.0.3",
    )

def repo_pypa_sphinxcontrib_serializinghtml():
    maybe(
        third_party_python_package,
        name = "pypa_sphinxcontrib_serializinghtml",
        target = "sphinxcontrib_serializinghtml",
        requirement = "sphinxcontrib-serializinghtml==1.1.4",
    )

def repo_pypa_toml():
    maybe(
        third_party_python_package,
        name = "pypa_toml",
        target = "toml",
        requirement = "toml==0.10.2",
    )

def repo_pypa_traitlets():
    repo_pypa_ipython_genutils()
    maybe(
        third_party_python_package,
        name = "pypa_traitlets",
        target = "traitlets",
        requirement = "traitlets==5.0.5",
        deps = [
            "@pypa_ipython_genutils//:ipython_genutils",
        ],
    )

def repo_pypa_typing_extensions():
    maybe(
        third_party_python_package,
        name = "pypa_typing_extensions",
        target = "typing_extensions",
        requirement = "typing-extensions==3.7.4.3",
    )

def repo_pypa_urllib3():
    maybe(
        third_party_python_package,
        name = "pypa_urllib3",
        target = "urllib3",
        requirement = "urllib3==1.26.2",
    )

def repo_pypa_wcwidth():
    maybe(
        third_party_python_package,
        name = "pypa_wcwidth",
        target = "wcwidth",
        requirement = "wcwidth==0.2.5",
    )

def repo_pypa_wheel():
    maybe(
        third_party_python_package,
        name = "pypa_wheel",
        target = "wheel",
        requirement = "wheel==0.36.2",
    )

def repo_pypa_zipp():
    maybe(
        third_party_python_package,
        name = "pypa_zipp",
        target = "zipp",
        requirement = "zipp==3.4.0",
    )
