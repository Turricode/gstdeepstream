plugin_c_args = ['-DHAVE_CONFIG_H']

cdata = configuration_data()
cdata.set_quoted('PACKAGE_VERSION', gst_version)
cdata.set_quoted('PACKAGE', 'gst-template-plugin')
cdata.set_quoted('GST_LICENSE', 'LGPL')
cdata.set_quoted('GST_API_VERSION', api_version)
cdata.set_quoted('GST_PACKAGE_NAME', 'GStreamer template Plug-ins')
cdata.set_quoted('GST_PACKAGE_ORIGIN', 'https://gstreamer.freedesktop.org')
configure_file(output : 'config.h', configuration : cdata)

gstsender_dep = dependency('gstreamer-1.0',
    fallback: ['gst-plugins-base'])

sender_sources = [
  'src/gstsender.c',
  ]

gstsender = library('gstsender',
  sender_sources,
  c_args: plugin_c_args,
  dependencies : [gst_dep, gstsender_dep],
  install : true,
  install_dir : plugins_install_dir,
)
