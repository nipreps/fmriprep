<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
<title></title>
<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">
<style type="text/css">
.sub-report-title {}
.run-title {}

h1 { padding-top: 35px; }
h2 { padding-top: 20px; }
h3 { padding-top: 15px; }

.elem-desc {}
.elem-filename {}

div.elem-image {
  width: 100%;
  page-break-before:always;
}

.elem-image object.svg-reportlet {
    width: 100%;
    padding-bottom: 5px;
}
body {
    padding: 65px 10px 10px;
}

.boiler-html {
    font-family: "Bitstream Charter", "Georgia", Times;
    margin: 20px 25px;
    padding: 10px;
    background-color: #F8F9FA;
}

div#boilerplate pre {
    margin: 20px 25px;
    padding: 10px;
    background-color: #F8F9FA;
}

</style>
</head>
<body>


<nav class="navbar fixed-top navbar-expand-lg navbar-light bg-light">
<div class="collapse navbar-collapse">
    <ul class="navbar-nav">
    {% for sub_report in sections %}
        {% if sub_report.isnested %}
        <li class="nav-item dropdown">
            <a class="nav-link dropdown-toggle" id="navbar{{ sub_report.name }}" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false" href="#">{{ sub_report.name }}</a>
            <div class="dropdown-menu" aria-labelledby="navbar{{ sub_report.name }}">
                {% for run_report in sub_report.reportlets %}
                <a class="dropdown-item" href="#{{run_report.name}}">{{run_report.title}}</a>
                {% endfor %}
            </div>
        </li>
        {% else %}
        <li class="nav-item"><a class="nav-link" href="#{{sub_report.name}}">{{sub_report.name}}</a></li>
        {% endif %}
    {% endfor %}
        <li class="nav-item"><a class="nav-link" href="#boilerplate">Methods</a></li>
        <li class="nav-item"><a class="nav-link" href="#errors">Errors</a></li>
    </ul>
</div>
</nav>
<noscript>
    <h1 class="text-danger"> The navigation menu uses Javascript. Without it this report might not work as expected </h1>
</noscript>

{% for sub_report in sections %}
    <div id="{{ sub_report.name }}">
    <h1 class="sub-report-title">{{ sub_report.name }}</h1>
    {% if sub_report.isnested %}
        {% for run_report in sub_report.reportlets %}
            <div id="{{run_report.name}}">
                <h2 class="run-title">Reports for {{ run_report.title }}</h2>
                {% for elem in run_report.reportlets %}
                    {% if elem.contents %}
                        {% if elem.title %}<h3 class="elem-title">{{ elem.title }}</h3>{% endif %}
                        {% if elem.description %}<p class="elem-desc">{{ elem.description }}<p>{% endif %}
                        {% for content in elem.contents %}
                            {% if elem.raw %}{{ content }}{% else %}
                            <div class="elem-image">
                            <object class="svg-reportlet" type="image/svg+xml" data="./{{ content }}">
                            Problem loading figure {{ content }}. If the link below works, please try reloading the report in your browser.</object>
                            </div>
                            <div class="elem-filename">
                                Get figure file: <a href="./{{ content }}" target="_blank">{{ content }}</a>
                            </div>
                            {% endif %}
                        {% endfor %}
                    {% endif %}
                {% endfor %}
            </div>
        {% endfor %}
    {% else %}
        {% for elem in sub_report.reportlets %}
            {% if elem.contents %}
                {% if elem.title %}<h3 class="elem-title">{{ elem.title }}</h3>{% endif %}
                {% if elem.description %}<p class="elem-desc">{{ elem.description }}<p><br />{% endif %}
                {% for content in elem.contents %}
                    {% if elem.raw %}{{ content }}{% else %}
                        <div class="elem-image">
                        <object class="svg-reportlet" type="image/svg+xml" data="./{{ content }}">filename:{{ content }}</object>
                        </div>
                        <div class="elem-filename">
                            Get figure file: <a href="./{{ content }}" target="_blank">{{ content }}</a>
                        </div>
                    {% endif %}
                {% endfor %}
            {% endif %}
        {% endfor %}
    {% endif %}
    </div>
{% endfor %}

<div id="boilerplate">
    <h1 class="sub-report-title">Methods</h1>
    {% if boilerplate %}
    <p>We kindly ask to report results preprocessed with fMRIPrep using the following
       boilerplate</p>
    <ul class="nav nav-tabs" id="myTab" role="tablist">
        {% for b in boilerplate %}
        <li class="nav-item">
            <a class="nav-link {% if b[0] == 0 %}active{% endif %}" id="{{ b[1] }}-tab" data-toggle="tab" href="#{{ b[1] }}" role="tab" aria-controls="{{ b[1] }}" aria-selected="{% if b[0] == 0 %}true{%else%}false{% endif %}">{{ b[1] }}</a>
        </li>
        {% endfor %}
    </ul>
    <div class="tab-content" id="myTabContent">
      {% for b in boilerplate %}
      <div class="tab-pane fade {% if b[0] == 0 %}active show{% endif %}" id="{{ b[1] }}" role="tabpanel" aria-labelledby="{{ b[1] }}-tab">{{ b[2] }}</div>
      {% endfor %}
    </div>
    {% else %}
    <p class="text-danger">Failed to generate the boilerplate</p>
    {% endif %}
    <p>Alternatively, an interactive <a href="http://fmriprep.readthedocs.io/en/latest/citing.html">boilerplate generator</a> is available in the <a href="https://fmriprep.org">documentation website</a>.</p>
</div>

<div id="errors">
    <h1 class="sub-report-title">Errors</h1>
    <ul>
    {% for error in errors %}
        <li>
        <div class="nipype_error">
            Node Name: <a target="_self" onclick="toggle('{{error.file|replace('.', '')}}_details_id');">{{ error.node }}</a><br>
            <div id="{{error.file|replace('.', '')}}_details_id" style="display:none">
            File: <code>{{ error.file }}</code><br>
            Working Directory: <code>{{ error.node_dir }}</code><br>
            Inputs: <br>
            <ul>
            {% for name, spec in error.inputs %}
                <li>{{ name }}: <code>{{ spec }}</code></li>
            {% endfor %}
            </ul>
            <pre>{{ error.traceback }}</pre>
            </div>
        </div>
        </li>
    {% else %}
        <li>No errors to report!</li>
    {% endfor %}
    </ul>
</div>


<script type="text/javascript">
    function toggle(id) {
        var element = document.getElementById(id);
        if(element.style.display == 'block')
            element.style.display = 'none';
        else
            element.style.display = 'block';
    }
</script>
</body>
</html>
