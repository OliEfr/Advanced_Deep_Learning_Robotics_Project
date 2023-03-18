# Setup

The implementations in this project are done with [Julia programming language](https://julialang.org/). Compared to Python, Julia has a much higher performance. This is especially useful for simulation and learning projects. Good interfaces to python are available. A disadvantage of Julia is, that it is a pretty new language and the community is still small.

Please refer to the official Julia documentation. This document provides a quickstart guide of using Julia for the project.

### Quickstart guide

    
1. [Download Julia](http://julialang.org/downloads/) (v1.8.2) I recommend to check for new versions!
    - For headless machines also download the current version from the website. apt-get may provide an old version.

    ```
    $ sudo apt install wget
    ```

    ```
    $ wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.2-linux-x86_64.tar.gz
    ```

    ```
    $ tar -xvzf julia-1.8.2-linux-x86_64.tar.gz
    ```


2. Install Julia  [Instructions for Official Binaries](https://julialang.org/downloads/platform/#platform_specific_instructions_for_official_binaries)

    - For Windows follow the installation instructions.
    - For Mac, copy Julia into your application folder.
    - For Linux, extract the folder and copy it to ``` /opt``` 

        ```
        $ sudo cp -r julia-1.8.2 /opt/
        ```

        and create a symbolic link to ```julia``` inside the ```/usr/local/bin``` folder:

        ```
        $ sudo ln -s /opt/julia-1.8.2/bin/julia /usr/local/bin/julia
        ```

        

3. Open Julia

    - For Windows by clicking on the new program icon on your desktop or where you specified it in the installation. Or [add the Julia path to your environment variables](https://www.geeksforgeeks.org/how-to-setup-julia-path-to-environment-variable/)
    - For Mac, by clicking on the new program icon.
    - Type ```julia``` in a new terminal.

4. A Julia terminal should open. To test, enter e.g. ``` sqrt(9)```. 
        
    Press ```ctrl + D``` if you want to close it again.


Then, do the following to use this repository:

5. Clone this repositoy 
6. Navigate into the cloned directory (using the terminal)

7.  ```$ julia``` to start julia
7.  ```julia> ]``` to open the Julia package manager.
8. ```(@v1.8) pkg> activate .``` The environment should be changed from ``` (@v1.7) pkg>  ``` to ```(Flyonic) pkg> ``` 
9. ```(Flyonic) pkg> instantiate``` This will take some time

in case there is a problem with electron [follow...](https://www.techomoro.com/how-to-install-and-set-up-electron-on-ubuntu-19-04-disco-dingo/)

10. Press ```ctrl + c``` to exit package manager
11. Now you can execute a RL-pipeline by using the `run_experiment.jl` file in the repository.

## Optional

To use the Jupiter notebooks in the example folder, IJulia must be installed in addition to Jupiter Notebook. To do this, go to the package manager in Julia again with ```]``` and write ```add IJulia```. Now Jupiter Notebook should be able to run the Julia examples.

We also use the tensorboard logger in our notebook. It can be installed by using ```add TensorBoardLogger```.


Depending on your personal taste, you can setup your favorite IDE. I recommend [Visual Studio Code](https://code.visualstudio.com/docs/languages/julia) with the [Julia extension](https://www.julia-vscode.org). But of course you can also choose from many other options Vim, Jupyter Notebook, Pluto Notebook, Atom, ...
