<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
<title></title>
<style type="text/css">
</style>
</head>
<body>



{% for sub_report in sub_reports %}
    <h2>{{ sub_report.name }}</h2>
    {% if sub_report.run_reports %}
        {% for run_report in sub_report.run_reports %}
            <h3>Reports for {{ run_report.title }}</h3>
            {% for elem in run_report.elements %}
                {% if elem.files_contents %}
                <h4>{{ elem.name }}<h4/>
                <br>
                {% for image in elem.files_contents %}
                    {{ image.1 }}<br>
                    {{ image.0 }}
                {% endfor %}
                {% endif %}
            {% endfor %}
        {% endfor %}
    {% else %}
        {% for elem in sub_report.elements %}
            {% if elem.files_contents %}
            <h4>{{ elem.name }}<h4/>
            <br>
            {% for image in elem.files_contents %}
                {{ image.1 }}<br>
                {{ image.0 }}
            {% endfor %}
            {% endif %}
        {% endfor %}

    {% endif %}
{% endfor %}

</body>
</html>
