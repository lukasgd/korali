Study Case: Reinforcement Learning for Prediction
==================================================

Environment for flow-prediction using reinforcement learning

The **state** is given the average value of the velocity and pressure and the respective values in the nearest neighbours blocks.

The **action** is given by the predicted next average value of the velocity and pressure.

The **reward** is given by least square error between the predicted flow and the actual flow.

Settings
--------

The setup is described in the setting.sh file.

Running the code
----------------

The application is launched locally (or on an interactive node on Piz Daint) using 

.. code-block:: bash

	./run-vracer-prediction.sh

If you are ready for production you can submit a job to the batch-system using

.. code-block:: bash

	./sbatch-run-vracer-prediction.sh

Here you can change the number of parallel agents by changing the NNODES variable.
