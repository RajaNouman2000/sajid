from django.urls import path
from . import views

urlpatterns = [
    path('hashtag-wordcloud/',views.hashtagworldcloud),
    path('user-engageement-metrics/',views.engagementmetrics),
    path('wordcloud/',views.worldcloud),
]
