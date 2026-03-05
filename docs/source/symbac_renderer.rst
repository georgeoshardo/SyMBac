SyMBac.renderer
==========================

.. autoclass:: SyMBac.renderer.Renderer
    :members:

    .. automethod:: __init__

Notes
-----

* Frame rendering now uses ``Renderer.render_frame(frame_index, config, ...)``
  with :class:`SyMBac.config_models.RenderConfig`.
* Dataset export is unified under
  ``Renderer.export_dataset(plan, output, base_config, ...)``.
* Interactive tuning is available with ``Renderer.create_tuner(...)`` and
  returns explicit configs via ``RenderTuner.current_config()``.
* See :doc:`migration_v1_api_cleanup` for old-to-new API mapping.
