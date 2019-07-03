Rails.application.routes.draw do
  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html
  #get '/uploads/new', to: 'uploads#new'
  #post '/uploads', to: 'uploads#create'

  resources :uploads
  resources :test
end
