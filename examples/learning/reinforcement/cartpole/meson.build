e = find_program('run-vracer.py', required: true)
test('learning.reinforcement.cartpole.continuous.single', e,
      args: [ '--concurrentEnvironments', '1'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-vracer.py', required: true)
test('learning.reinforcement.cartpole.continuous.multiple', e,
      args: [ '--concurrentEnvironments', '2'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
    
e = find_program('run-dvracer.py', required: true)
test('learning.reinforcement.cartpole.discrete.single', e,
      args: [ '--concurrentEnvironments', '1'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-dvracer.py', required: true)
test('learning.reinforcement.cartpole.discrete.multiple', e,
      args: [ '--concurrentEnvironments', '2'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

