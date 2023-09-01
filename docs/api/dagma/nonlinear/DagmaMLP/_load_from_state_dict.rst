:py:meth:`dagma.nonlinear.DagmaMLP._load_from_state_dict <dagma.nonlinear.DagmaMLP._load_from_state_dict>`
==========================================================================================================
.. _dagma.nonlinear.DagmaMLP._load_from_state_dict:
.. py:method:: dagma.nonlinear.DagmaMLP._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

   Copies parameters and buffers from :attr:`state_dict` into only
   this module, but not its descendants. This is called on every submodule
   in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
   module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
   For state dicts without metadata, :attr:`local_metadata` is empty.
   Subclasses can achieve class-specific backward compatible loading using
   the version number at `local_metadata.get("version", None)`.

   .. note::
       :attr:`state_dict` is not the same object as the input
       :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
       it can be modified.

   :param state_dict: a dict containing parameters and
                      persistent buffers.
   :type state_dict: dict
   :param prefix: the prefix for parameters and buffers used in this
                  module
   :type prefix: str
   :param local_metadata: a dict containing the metadata for this module.
                          See
   :type local_metadata: dict
   :param strict: whether to strictly enforce that the keys in
                  :attr:`state_dict` with :attr:`prefix` match the names of
                  parameters and buffers in this module
   :type strict: bool
   :param missing_keys: if ``strict=True``, add missing keys to
                        this list
   :type missing_keys: list of str
   :param unexpected_keys: if ``strict=True``, add unexpected
                           keys to this list
   :type unexpected_keys: list of str
   :param error_msgs: error messages should be added to this
                      list, and will be reported together in
                      :meth:`~torch.nn.Module.load_state_dict`
   :type error_msgs: list of str



