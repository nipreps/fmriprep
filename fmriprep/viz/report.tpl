<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
<title></title>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">
<style type="text/css">
.sub-report-title {}
.run-title {}
.elem-title {}
.elem-desc {}
.elem-filename {}
.elem-image svg {
    width: 100%;   
}
body { 
    padding-top: 65px; 
}
</style>
</head>
<body>

<nav class="navbar navbar-default navbar-fixed-top">
<div class="container collapse navbar-collapse">
    <ul class="nav navbar-nav">
        {% for sub_report in sub_reports %}
            {% if sub_report.run_reports %}
                <li class="dropdown">
                    <a class="nav-item  nav-link dropdown-toggle" type="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="true" href="">
                        {{ sub_report.name }}
                        <span class="caret"></span>
                    </a>
                    <ul class="dropdown-menu">
                    {% for run_report in sub_report.run_reports %}
                        <li><a class="dropdown-item" href="#{{run_report.name}}">{{run_report.title}}</a></li>
                    {% endfor %}
                    <li><a class="dropdown-item" href="#errors">Errors</a></li>
                    </ul>
                </li>
            {% else %}
                <li><a href="#{{sub_report.name}}">{{ sub_report.name }}</a></li>
            {% endif %}
        {% endfor %}
    </ul>
<div>
</nav>
<noscript>
    <h1 class="text-danger"> The navigation menu uses Javascript. Without it this report might not work as expected </h1>
</noscript>

{% for sub_report in sub_reports %}
    <div id="{{ sub_report.name }}">
    <h1 class="sub-report-title">{{ sub_report.name }}</h1>
    {% if sub_report.run_reports %}
        {% for run_report in sub_report.run_reports %}
            <div id="{{run_report.name}}">
                <h2 class="run-title">Reports for {{ run_report.title }}</h2>
                {% for elem in run_report.elements %}
                    {% if elem.files_contents %}
                        <h3 class="elem-title">{{ elem.title }}</h3>
                        <p class="elem-desc">{{ elem.description }}<p>
                        <br>
                        {% for image in elem.files_contents %}
                            <div class="elem-image">{{ image.1 }}</div><br>
                            <div class="elem-filename">
                                Filename: {{ image.0 }}
                            </div>
                        {% endfor %}
                    {% endif %}
                {% endfor %}
            </div>
        {% endfor %}
    {% else %}
        {% for elem in sub_report.elements %}
            {% if elem.files_contents %}
            <h3 class="elem-title">{{ elem.title }}</h3>
            <p class="elem-desc">{{ elem.description }}<p>
            <br>
            {% for image in elem.files_contents %}
                <div class="elem-image">{{ image.1 }}</div><br>
                Filename: {{ image.0 }}
            {% endfor %}
            {% endif %}
        {% endfor %}
    {% endif %}
    </div>
{% endfor %}

<div id="errors">
    <h1 class="sub-report-title">Errors</h1>
    <ul>
    {% for error in errors %}
        <li>
        <div class="nipype_error">
            Node Name: <a target="_self" onclick="toggle('{{error.file|replace('.', '')}}_details_id');">{{ error.node }}</a><br>
            <div id="{{error.file|replace('.', '')}}_details_id" style="display:none">
            File: {{ error.file }}<br>
            Working Directory: {{ error.node_dir }}<br>
            Inputs: <br>
            <ul>
            {% for name, spec in error.inputs %}
                <li>{{ name }}: {{ spec }}</li>
            {% endfor %}
            </ul>
            <pre>
            {{ error.traceback }}
            </pre>
            </div>
        </div>
        </li>
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
