# Trip Price Prediction

Use regression in pyspark to predict optimal price for each trip

Tip: If you don't have a markdown viewer like atom, you can render this on chrome by following [this link](https://imagecomputing.net/damien.rohmer/teaching/general/markdown_viewer/index.html).
# Pre-requisites

* Ensure you have `Miniconda` installed and can be run from your shell. If not, download the installer for your platform here: https://docs.conda.io/en/latest/miniconda.html

     **NOTE**

     * If you already have `Anaconda` installed, go ahead with the further steps, no need to install miniconda.
     * If `conda` cmd is not in your path, you can configure your shell by running `conda init`.


* Ensure you have `git` installed and can be run from your shell

     **NOTE**

     * If you have installed `Git Bash` or `Git Desktop` then the `git` cli is not accessible by default from cmdline.
       If so, you can add the path to `git.exe` to your system path. Here are the paths on a recent setup

```
        %LOCALAPPDATA%\Programs\Git\git-bash.exe
        %LOCALAPPDATA%\GitHubDesktop\app-<ver>\resources\app\git\mingw64\bin\git.exe
```

* Ensure [invoke](http://www.pyinvoke.org/index.html) tool and pyyaml are installed in your `base` `conda` environment. If not, run

```
(base):~$ pip install invoke
(base):~$ pip install pyyaml
```

# Getting started

* Switch to the root folder (i.e. folder containing this file)
* A collection of workflow automation tasks can be seen as follows

```
(base):~/<proj-folder>$ inv -l
```

* To verify pre-requisites, run

```
(base)~/<proj-folder>$ inv debug.check-reqs
```

and check no error messages (`Error: ...`) are printed.


## Environment setup:

### Introduction
* Environment is divided into two sections

    * Core - These are must have packages & will be setup by default. These are declared in `deploy/pip/ct-pyspark-dev.txt` and `ct-core-dev.txt`.
    * Addons - These are for specific purposes you can choose to install. Here are the addon options
        * `formatting` - To enforce coding standards in your projects.
        * `documentation` - To auto-generate doc from doc strings and/or create rst style documentation to share documentation online
        * `testing` - To use automated test cases
        * `jupyter` - To run the notebooks. This includes jupyter extensions for spell check, advances formatting.
        * `extras` - there are nice to haves or for pointed usage.
        * `ts` - Install this to work with time series data
        * `pyspark` - Installs pyspark related dependencies in the env.
    * Edit the addons here `deploy/pip/addon-<addon-name>-dev.txt` to suit your need.
    * Each of the packages there have line comments with their purpose. From an installation standpoint extras are treated as addons
* You can edit them to your need. All these packages including addons & extras are curated with versions & tested throughly for acceleration.
* While you can choose, please decide upfront for your project and everyone use the same options.
* Below you can see how to install the core environment & addons separately. However, we strongly recommend to update the core env with the addons packages & extras as needed for your project. This ensures there is only one version of the env file for your project.
* **To run the reference notebooks and production codes, it is recommended to install all addons.**
* Tip: Default name of the env is `ta-lib-pyspark-dev`. You can change it for your project.
    * For example: to make it as `env-myproject-prod`.
    * Open `tasks.py`
    * Set `ENV_PREFIX_PYSPARK = 'env-customer-x'`

### Setup a development environment:
```
(base):~/<proj-folder>$ inv dev.setup-env-pyspark
```

The above command should create a conda python environment named `ta-lib-pyspark-dev` (by default).

You also have the option to create a conda environment with a specific python version. 
The below command will a create conda environment with python version `3.9`.
```
(base):~/<proj-folder>$ inv dev.setup-env-pyspark --python-version=3.9
```
`python-version` parameter above is an optional parameter. By default it is set to `3.10` but it can take values of `3.8`, `3.9` or `3.10`.

* Activate the environment:

Activate the environment first to install other addons. Keep the environment active for all the remaining commands in the manual.
```
(base):~/<proj-folder>$ conda activate ta-lib-pyspark-dev
```

Install `invoke` and `pyyaml` in this env to be able to install the addons in this environment.
```
(ta-lib-pyspark-dev):~/<proj-folder>$ pip install invoke
```
```
(ta-lib-pyspark-dev):~/<proj-folder>$ pip install pyyaml
```

Now run all following command to install all the addons. Feel free to customize addons as suggested in the introduction.

```
(ta-lib-pyspark-dev):~/<proj-folder>$ inv dev.setup-addon-pyspark --formatting --jupyter --documentation --testing --extras --ts
```

## Manual Environment Setup

If you are facing difficulties setting up the environment using the automated process (i.e., using the `invoke command`) 
or if the command is not accessible, you can use these manual steps.  This approach is 
particularly useful when troubleshooting or in situations where automated setup is not feasible.
Follow the steps below to manually set up the environment.

### Step 1: Create virtual Environment
```
(base):~/<proj-folder>$ conda create --name <env_name> python=<python_version>
```
Replace `<env_name>` with your specific environment name, and `<python_version>` with the desired python version (e.g., `3.8`, `3.9`, `3.10`).

### Step 2: Activate the Environment
```
(base):~/<proj-folder>$ conda activate <env_name>
```

### Step 3: Install Core Packages
```
(<env_name>):~/<proj-folder>$ pip install -r deploy/pip/ct-core-dev.txt
```

### Step 4: Install the ta_lib editable package
```
(<env_name>):~/<proj-folder>$ pip install -e <path_to_setup.py>
```
if you are in the same level as the `setup.py` file, you can use:
```
(<env_name>):~/<proj-folder>$ pip install -e .
```

### Step 5: Install Additional Packages for Pyspark
```
(<env_name>):~/<proj-folder>$ pip install -r deploy/pip/ct-pyspark-dev.txt
```

### Step 6 (Optional): Install Additional Addons
```
(<env_name>):~/<proj-folder>$ pip install -r <path_to_addon_file>
```
For example, to install `jupyter` addons, use the following command:
```
(<env_name>):~/<proj-folder>$ pip install -r deploy/pip/addon-jupyter-dev.txt
```

## Setting Up Environment in Cloud
In a cloud environment invoke commands may not be accessible.
To set up the environment in a cloud setting, you can refer to the following link: [Cloud Environment Setup](https://tigeranalytics-code-templates.readthedocs-hosted.com/en/latest/code_templates/installation_setup.html)

# Launch a Jupyter notebook within this environment to start using Spark:
- In order to launch a jupyter notebook locally in the web server, run
```
(ta-lib-pyspark-dev):~/<proj-folder>$ inv launch.jupyterlab-pyspark
```
    After running the command, type [localhost:8080](localhost:8080) to see the launched JupyterLab.


- The `inv` command has a built-in help facility available for each of the invoke builtins. To use it, type `--help` followed by the command:
    ```
    (ta-lib-pyspark-dev):~/<proj-folder>$ inv launch.jupyterlab-pyspark --help
    ```
- On running the ``help`` command, you get to see the different options supported by it.

```
    Usage: inv[oke] [--core-opts] launch.jupyterlab-pyspark [--options] [other tasks here ...]

    Options:
    -a STRING, --password=STRING
    -e STRING, --env=STRING
    -i STRING, --ip=STRING
    -o INT, --port=INT
    -p STRING, --platform=STRING
    -t STRING, --token=STRING
```

# Using customer docker containers in Databricks

This template provides an overview of getting started with pyspark classification and regression notebooks with Databricks clusters using [`databricks-connect`](https://docs.databricks.com/dev-tools/databricks-connect.html).

* Install docker: https://docs.docker.com/docker-for-windows/install/

* Docker Hub:

    * Create a Docker account: https://hub.docker.com
    * Create a private/public repository (Docker Hub provides the default flexibility to create one private repository)

* Amazon ECR:
    * [Create](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) your ECR repository 
    * [Create your access keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html) by expanding the Access Keys (access key ID and secret access key) section under: 
    
            User -> My Security Credentials 
        
    * Also ensure you have AWS CLI installed: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-windows.html

## Docker container setup
* Build the docker image:

```
(ta-lib-pyspark-dev):~$ cd deploy/pyspark/
(ta-lib-pyspark-dev):~$ docker build -t <repository_name>:<tag> .    # example : docker build -t helloapp:v1 .
```

* Verify the installation by listing the installed images

```
(ta-lib-pyspark-dev):~$ docker images
```

* Push the docker image to your repository:
    * DockerHub
    
    ```
    (ta-lib-pyspark-dev):~$ docker login --username <username>            # enter the password as prompted
    (ta-lib-pyspark-dev):~$ docker tag <image> <username>/<repo-name>     # example : docker tag 88ddeec9d217 amritbhaskar/ta_lib
    (ta-lib-pyspark-dev):~$ docker push <username>/<repo-name>            # example : docker push amritbhaskar/ta_lib
    ```

    * ECR:
    ```
    (ta-lib-pyspark-dev):~$ aws configure                                                                      # enter the AWS Access Key ID & AWS Secret Access Key for the repository as prompted
    (ta-lib-pyspark-dev):~$ aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.region.amazonaws.com/<my-repo>
    (ta-lib-pyspark-dev):~$ docker tag <image> <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<repo-name>     # example: docker tag 88ddeec9d217 aws_account_id.dkr.ecr.us-east-1.amazonaws.com/ta_lib
    (ta-lib-pyspark-dev):~$ docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<repo-name>
    ```




## Using your docker container 

Refer to the [Databricks Container Services documentation](https://docs.databricks.com/clusters/custom-containers.html) for additional details.

* Login to the databricks workspace 

* Enable container services in the Databricks account as follows: 

        Admin Console -> Advanced -> Container Services

* [Launch](https://docs.databricks.com/clusters/create.html) your cluster using the UI

* Select the **Use your own Docker container** option
 
* Enter your custom Docker image in the Docker Image URL field and give the docker repository details.

    Example - 

        DockerHub:
            <dockerid>/<repository>:<tag>
            amritbhaskar/ta_lib:latest
            
        Elastic Container Registry:
            <aws-account-id>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>

    * Select the relevant authentication type 

        * Default - When using a public Docker repository or when using Default mode with an IAM role in ECR.
        * Username and password - Provide username and password  in case you are using a DockerHub private repository 

    * Confirm the changes and restart the cluster.

### Adding packages

* For installation of any package while working on the notebook:
```
! pip install {packagename}
```

* To add new packages to docker container.

    * Add the package to the `env.yml` file.
    * Rerun the command ```docker build -t [Repository-name]:[tag]```  to rebuild the repository.
    * Push the image to the DockerHub/ECR.
    
## Databricks Connect setup

You can use Databricks Connect to run Spark jobs from your notebook apps or IDEs. The Databricks Connect package is already installed as part ofg the `ta-lib-dev-pyspark` environment.

* Login to the Databricks workspace 

* Navigate to your required cluster, and ensure that your Spark Config is set to ```spark.databricks.service.server.enabled true``` under 

        Advanced Options -> Spark -> Spark Config
        
* In your account, generate a new token under

        User Settings -> Access Tokens -> Generate New Token
    Copy this token and save it for further use.
        
* Collect the following [details](https://docs.databricks.com/workspace/workspace-details.html) regarding your workspace:
    * Databricks host (e.g. ```https://dbc-0b606se5-478a.cloud.databricks.com/?o=4234485324337931```)
    * Databricks token (saved from the previous step)
    * Cluster ID (e.g.0815-092137-flare283)
    * Org ID
    * Port (15001)
    
* Check the version of Python running (should be 3.7 for the above cluster)

* Check that you have [Java SE Runtime Version 8](https://www.oracle.com/java/technologies/javase-jre8-downloads.html) installed and added in your system environment variables
    
* Run the cmd: ```databricks-connect configure``` and enter the saved information when prompted


    
# Frequently Asked Questions

The FAQ for code templates during setting up, testing, development and adoption phases are available
[here](https://tigeranalytics-code-templates.readthedocs-hosted.com/en/latest/faq.html)