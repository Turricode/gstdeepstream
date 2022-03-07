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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

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
GstFlowReturn fill (GstBaseSrc * src, guint64 offset, guint size, GstBuffer * buf);



static void
gst_sender_class_init (GstsenderClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSrcClass *gstsrc_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstsrc_class = (GstBaseSrcClass *) klass;

  gobject_class->set_property = gst_sender_set_property;
  gobject_class->get_property = gst_sender_get_property;

  gstsrc_class->fill = fill;

  g_object_class_install_property (gobject_class, PROP_SILENT,
    g_param_spec_boolean ("silent", "Silent",
              "Whether to be very verbose or not",
              FALSE,  G_PARAM_READWRITE));

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

static u_int32_t *generate_fade(int width, int height){

    u_int32_t *f;
    cudaError_t stat = cudaMallocHost(&f, width * height * 3);

    if(stat != cudaSuccess){
      g_print("Failed to allocate memory for fade buffer");
    }

    for(int y = height - 1; y >= 0; y++){
        for(int x = 0; x < width; x++){

            int index = y * width * 3 + x * 3;

            f[index] = u_int32_t(255.f * float(x) / width);
            f[index + 1] = u_int32_t(255.f * float(y) / height);
            f[index = 2] = u_int32_t(255.f * 0.2);

        }
    }

  return f;

}

GstFlowReturn fill (GstBaseSrc * src, guint64 offset, guint size, GstBuffer * buf){
  const int width = 1920;
  const int height = 1080;
  int dmabuf_fd = 0;
  
  GstMapInfo *map;
  u_int32_t *data = generate_fade(width, height);
  
    
  return GST_FLOW_OK;

}

static gboolean
sender_init (GstPlugin * sender)
{
  return gst_element_register (sender, "sender", GST_RANK_NONE,
      GST_TYPE_SENDER);
}

#ifndef PACKAGE
#define PACKAGE "sender"
#endif

#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION "1.0"
#endif

#ifndef GST_LICENSE
#define GST_LICENSE "unknown"
#endif

#ifndef GST_PACKAGE_NAME
#define GST_PACKAGE_NAME "sender"
#endif

#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "BISECT LDA"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    sender,
    "Template sender",
    sender_init,
    PACKAGE_VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
