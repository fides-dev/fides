.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes Summary

.. autosummary::
    {% for class in classes %}
    {{ class }}
    {% endfor %}

{% endif %}

{% if functions %}
.. rubric:: Functions Summary

.. autosummary::
    {% for function in functions %}
    {{ function }}
    {% endfor %}

{% endif %}

{% if classes %}
.. rubric:: Classes

{% for class in classes %}
.. autoclass:: {{ class }}
    :members:
    :special-members: __init__
{% endfor %}

{% endif %}

{% if functions %}
.. rubric:: Functions

{% for function in functions %}
.. autofunction:: {{ function }}
{% endfor %}

{% endif %}