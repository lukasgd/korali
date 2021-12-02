Study Case: Reinforcement Learning for Artificial Swimmers
===========================================================

Environment for artificial swimmers, described by the StefanFish class (see _deps/CUP2D/Obstacles/Stefanfish{.cpp,.h}).

The **state** is given the relative position :math:`\Delta x, \Delta y` to the obstacle, the orientation :math:`\theta`, the phase :math:`\phi`, the speed :math:`u,v` and angular velocity :math:`\omega`, the time of the last action :math:`t_{act}` and previous two baseline curvatures :math:`C_{-1},C_{-2}`, as well as the shear stress at three sensor locations :math:`\tau_1,\tau_2,\tau_3`.

The **action** is given by modifying the baseline curvature :math:`C_0` and the swimming period :math:`T`.

The **reward** is given by the (Froude) swimming efficiency :math:`\eta`.

Settings
--------

The setup is described in the setting.sh file.

Running the code
----------------

The application is launched locally (or on an interactive node on Piz Daint) using 

.. code-block:: bash

	./run-vracer-swimmer.sh

If you are ready for production you can submitt a job to the batch-system using

.. code-block:: bash

	./sbatch-run-vracer-swimmer.sh

Here you can change the number of parallel agents by changing the NNODES variable.
