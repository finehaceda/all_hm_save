from django.urls import path

from polls import views

urlpatterns = [
    # ex: /polls/
    # path("", views.index, name="index"),
    # # ex: /polls/5/
    # path("<int:question_id>/", views.detail, name="detail"),
    # # ex: /polls/5/results/
    # path("<int:question_id>/results/", views.results, name="results"),
    # # ex: /polls/5/vote/
    # path("<int:question_id>/vote/", views.vote, name="vote"),
    # # path("encodetest", views.dna_encoding_view, name="encode"),
    # path('submit/', views.my_view, name='submit_form'),

    path("encode", views.dna_encoding, name="encode"),

    # path('encode_to_synseq', views.encode_to_synseq, name='encode_to_synseq'),
    path('simulate', views.simulate_view, name='simulate'),
    # path('seqencing_to_cluster', views.seq_to_cluster, name='seqencing_to_cluster'),
    # path('cluster', views.cluster_view, name='cluster'),
    path('reconstruct', views.reconstruct_view, name='reconstruct'),
    path('cluster_to_decode', views.decode_view, name='cluster_to_decode'),
    path('decode', views.decode_view, name='decode'),
    path('evaluate', views.evaluate_view, name='evaluate'),
    # path('download', views.download_file, name='download_file'),
    path('download/<str:mode>/', views.download_file, name='download_file'),
    # path('test', views.test, name='test'),

    # path('execute-view/', views.execute_view, name='execute_view'),

]