//
//  LoginView.swift
//  Tridify
//
//  Created by Maged Alosali on 28/04/2023.
//

import SwiftUI

struct LoginView: View {
    
    @Environment(\.colorScheme) private var colorScheme
    private var isLightMode: Bool {
        colorScheme == .light
    }
    
    @State private var email = ""
    @State private var password = ""
    @State private var showPassword = false
    
    var body: some View {
        GeometryReader { geo in
            VStack {
                LogoView()
                    .frame(width: geo.size.width*0.275)
                    .frame(width: geo.size.width)
                Text ("LOGIN")
                    .font(.title)
                    .kerning(10)
                Text("Get started to save your models to the web and access them on any device.")
                    .font(.headline)
                    .foregroundColor(isLightMode ? .lightHeadline : .darkHeadline)
                    .fontWeight(.light)
                    .multilineTextAlignment(.center)
                    .frame(width: geo.size.width * 0.75)
                    .padding(.top, 0.1)
    
                VStack {
                    
                    VStack (alignment: .leading) {
//                        Label("Email ", systemImage: "envelope.fill")
//                            .font(.title2)
                        Text("Email")
                            .font(.title2)
                        TextField("Enter your email address", text: $email)
                            .padding(.horizontal, 20)
                            .font(.title3)
                            .overlay(content: {
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke()
                                    .frame(height: 35)
                                    .foregroundColor(.gray)
                            })
                    }
                    
                    VStack(alignment: .leading) {
//                        Label("Password ", systemImage: "lock.square")
//                            .font(.title2)
                        Text("Password")
                            .font(.title2)
                        Group {
                            if !showPassword {
                                SecureField("Enter your password", text: $password)
                            }
                            else {
                                TextField("Enter your password", text: $password)
                            }
                        }
                        .padding(.horizontal, 20)
                        .font(.title3)
                        .overlay(content: {
                            RoundedRectangle(cornerRadius: 10)
                                .stroke()
                                .frame(height: 35)
                                .foregroundColor(.gray)
                        
                        })
                        .overlay(alignment: .trailing) {
                            Button {
                                showPassword.toggle()
                            } label: {
                                Image(systemName: showPassword ? "eye" : "eye.slash")
                                    .foregroundColor(showPassword ? .link : .gray)
                                    .padding(.horizontal, 6)
                            }
                        }
                        
                        NavigationLink {
                            Text("Forgot password View")
                                
                        } label: {
                            Text("Forgot password ?")
                                .foregroundColor(.link)
                                .font(.callout)
                        }
                        .padding([.horizontal, .top], 5)
                    }
                    .padding(.top)
                    
                    
                }
                .padding([.top, .horizontal], 15)
                
                Spacer()
                ButtonOneView(buttonText: "Login", systemName: nil, targetView: {
                    Text("Main View")
                })
                .frame(width: geo.size.width * 0.7)
                
                AccountStatusView(questionText: "Dont have an account?", navigationText: "Sign Up", targetView: {
                    Text("Sign up view")
                })
                .padding(.top, 3)
                
                Spacer()
                
                
            }
        }
    }
}

struct LoginView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            LoginView()
            
        }
    }
}
