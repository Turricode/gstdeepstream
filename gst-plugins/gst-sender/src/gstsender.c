/**
 * SECTION:element-sender
 *
 * FIXME:Describe sender here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! sender ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/base/base.h>
#include <gst/controller/controller.h>

#include "gstsender.h"

GST_DEBUG_CATEGORY_STATIC (gst_sender_debug);
#define GST_CAT_DEFAULT gst_sender_debug

enum
{
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT,
};

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("ANY")
    );

#define gst_sender_parent_class parent_class
G_DEFINE_TYPE (Gstsender, gst_sender, GST_TYPE_SENDER);

static void gst_sender_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_sender_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);


static void
gst_sender_class_init (GstsenderClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_sender_set_property;
  gobject_class->get_property = gst_sender_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE | GST_PARAM_CONTROLLABLE));

  gst_element_class_set_details_simple (gstelement_class,
      "sender",
      "Generic/Filter",
      "FIXME:Generic Template Filter", "bisect <<user@hostname.org>>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_template));

  GST_DEBUG_CATEGORY_INIT (gst_sender_debug, "sender", 0,
      "Template sender");
}


static void
gst_sender_init (Gstsender * filter)
{
  filter->silent = FALSE;
}

static void
gst_sender_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gstsender *filter = GST_SENDER(object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_sender_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gstsender *filter = GST_SENDER (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
sender_init (GstPlugin * sender)
{
  return gst_element_register (sender, "sender", GST_RANK_NONE,
      GST_TYPE_SENDER);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    sender,
    "Template sender",
    sender_init,
    PACKAGE_VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
