#ifndef __GST_SENDER_H__
#define __GST_SENDER_H__

#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>

G_BEGIN_DECLS

#define GST_TYPE_SENDER (gst_sender_get_type())
G_DECLARE_FINAL_TYPE (Gstsender, gst_sender,
    GST, SENDER, GstBaseSrc)

struct _Gstsender {
  GstBaseSrc element;

  gboolean silent;
};

G_END_DECLS

#endif /* __GST_SENDER_H__ */
