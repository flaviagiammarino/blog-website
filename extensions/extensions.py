def google_analytics(app, pagename, templatename, context, doctree):
    metatags = context.get('metatags', '')
    metatags += """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-LPMJ6R8T4C"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
    
      gtag('config', 'G-LPMJ6R8T4C');
    </script>
    """
    context['metatags'] = metatags


def on_builder_inited(app):
    if app.builder.name == 'html':
        app.builder.search = False
        app.builder.use_index = False


def setup(app):
    app.connect('html-page-context', google_analytics)
    app.connect('builder-inited', on_builder_inited)
    